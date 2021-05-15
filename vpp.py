from generic import iter_collect_files
from monitors import FileMonitor, REMOVED, MODIFIED
from inspect import getmembers, isfunction
from ast import literal_eval
from pathlib import Path
from mako.template import Template
from mako import exceptions
from mako.lookup import TemplateLookup

# from mako import exceptions
import logging
import time
import re
import os
import yaml
import sys
import filters
import argparse


class ComputedError(Exception):
    def __init__(self, msg):
        super(ComputedError, self).__init__(
            f"failed to evaluate computed property: {msg}"
        )


class ExtensionError(Exception):
    def __init__(self, msg):
        super(ExtensionError, self).__init__(
            f"filename should always end with .tpp, invalid filename: {msg}"
        )


def tpp_preprocessor(source: str) -> str:
    """
    tpp do folloings
    1. escape ${} for tcl, bash ...
    2. disable newline filter
    3. change substituion syntax ${expr} to @{{expr}}
    """
    source = re.sub(
        r"\${(.+?)}", r"${'${'}\1${'}'}", source, flags=re.DOTALL
    )  # escape ${}
    source = re.sub(
        r"\@{{(.+?)}}", r"${\1}", source, flags=re.DOTALL
    )  # map @{{epxr}} back to ${expr}
    source = re.sub(r"\\", r"${'\\\'}", source)  # disable newline filter
    return source


def parse_args() -> argparse.Namespace:
    def path_wrapper(pathname: str) -> Path:
        if not pathname:
            return None
        path = Path(os.path.expandvars(pathname)).expanduser().resolve()
        if path.is_file() and path.suffix != ".tpp":
            raise ExtensionError(path)
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pathnames",
        metavar="template or directory",
        nargs="*",
        type=path_wrapper,
        default=[],
    )
    parser.add_argument(
        "-g",
        "--globals",
        metavar="YAML_FILE",
        default=None,
        type=path_wrapper,
        help="a YAML file to specify global variables",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        default="./",
        type=path_wrapper,
        help="specify a directory to dump output. default: CWD",
    )
    parser.add_argument(
        "-f",
        "--filelist",
        action="append",
        default=[],
        type=path_wrapper,
        help="a filelist of templates",
    )
    parser.add_argument(
        "--incdir",
        default=[],
        metavar="DIRECTORY",
        action="append",
        type=path_wrapper,
        help="specify a directory to lookup templates, which won't be rendered automatically.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="running at development mode, which will render any template if it had been modified",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="show more informations",
    )
    return parser.parse_args()


def render(
    template: Path,
    output_directory: Path,
    _globals: dict,
    logger: logging.Logger = None,
    lookup: TemplateLookup = None,
):
    if not template.suffix == ".tpp":
        raise ExtensionError(template)

    try:
        tmp = Template(
            filename=str(template),
            preprocessor=tpp_preprocessor,
            lookup=lookup,
        )

        out = output_directory.joinpath(template.stem)
        msg = f"{'generated'} | {out}"
        with out.open("w") as fo:
            fo.write(tmp.render(store={}, **_globals))
            logger.info(msg) if logger else print(msg)
    except:
        sep = "\n  "
        errors = ["  "]
        traceback = exceptions.RichTraceback()
        for (filename, lineno, function, line) in traceback.traceback:
            # if filename == str(template):
            errors.append("File %s, line %s, in %s" % (filename, lineno, function))
            errors.append(line)
        errors.append(
            "%s: %s" % (str(traceback.error.__class__.__name__), traceback.error)
        )
        msg = sep.join(errors)
        logger.error(msg) if logger else print(msg)


def main():
    argv = parse_args()

    ### setup a root logger
    sh = logging.StreamHandler()
    fh = logging.FileHandler(f"{__name__}.log", mode="w")
    logging.basicConfig(
        level=logging.INFO if argv.verbose else logging.ERROR,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[sh, fh],
    )
    root_logger = logging.getLogger("TPP")

    ### setup global variables for template engine
    _globals = dict(getmembers(filters, isfunction))
    if argv.globals:
        with argv.globals.open("r") as fo:
            config = yaml.load(fo, Loader=yaml.SafeLoader)

        _globals.update(config.get("globals", {}))
        computed = config.get("computed", {})

        for var, expr in computed.items():
            temp = Template("${%s}" % expr)
            try:
                value = temp.render(**_globals)
            except Exception as e:
                raise ComputedError(f"{var}: {expr}")
            try:
                # eval a integer, list or dict ...
                _globals[var] = literal_eval(value)
            except ValueError:
                # the value must be a normal string
                _globals[var] = value

    ### create output directories
    output_directory = argv.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    sources = argv.filelist + argv.pathnames
    ### render templates
    lookup = (
        TemplateLookup(
            directories=["lookup/"],
            preprocessor=tpp_preprocessor,
            module_directory="./tmp",
        )
        if argv.incdir
        else None
    )
    x = lookup.get_template("header.txt")
    print(x)
    for f in iter_collect_files(*sources):
        # lookup.put_template(
        #     f.name,
        #     template=Template(
        #         module_directory="./tmp",
        #         filename=str(f.relative_to(Path.cwd())),
        #         preprocessor=tpp_preprocessor,
        #         lookup=lookup,
        #     ),
        # )
        lookup.put_string(f.name, f.open().read())
    print(lookup._collection)

    def render_all():
        for filename in iter_collect_files(*sources):
            # render(
            #     filename, output_directory, _globals, lookup=lookup, logger=root_logger
            # )
            x = lookup.get_template(filename.name)
            print(x)
            x.render(store={}, **_globals)

    render_all()  # always render all once

    if argv.dev:
        filemonitor = FileMonitor(*sources)
        lookup_monitor = FileMonitor(*argv.incdir)
        print('Running at development mode. Use "ctrl+c" to exit.')
        while True:
            try:
                for event, _ in lookup_monitor.iter_diff(verbose=False):
                    if event != REMOVED:
                        render_all()
                        break
                for event, filename in filemonitor.iter_diff(verbose=False):
                    root_logger.info(f"{event} | {filename}")
                    if event != REMOVED:
                        continue
                    render(
                        Path(filename), output_directory, _globals, logger=root_logger
                    )
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nBye :D\n")
                return


if __name__ == "__main__":
    sys.exit(main())
