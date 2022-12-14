from typing import Tuple, List, Union
from .generic import iter_collect_files
from .monitors import FileMonitor, REMOVED, ImportedModulesMonitor
from inspect import getmembers, isfunction
from ast import literal_eval
from pathlib import Path
from mako.template import Template
from mako import exceptions
from mako.lookup import TemplateLookup
from .vpp.vpp import VerilogModule, tie0, tie_max, ASSIGN, TIE

from . import filters

import logging
import time
import re
import os
import yaml
import argparse
import tempfile


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


def parse_args(args=None) -> argparse.Namespace:
    def path_wrapper(pathname: str) -> Path:
        if not pathname:
            return None
        path = Path(os.path.expandvars(pathname)).expanduser().resolve()
        return path

    def remap_define(string: str) -> List[Tuple[str, Union[str, int]]]:
        defines = []
        for define in string.split("+"):
            if not define:
                continue
            define = define.split("=")
            if len(define) == 2:
                defines.append(tuple(define))
            elif len(define) == 1:
                defines.append((define[0], ""))
            else:
                raise SyntaxError(define)
        return defines

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pathnames",
        metavar="file or directory",
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
        "-d",
        "--define",
        default=[],
        action="extend",
        type=remap_define,
        metavar="DEFINE",
        dest="defines",
        help="specify global defines for verilog preprocessor",
    )
    parser.add_argument(
        "--import",
        default=[],
        metavar="PYTHON FILE or PACKAGE",
        action="append",
        type=str,
        dest="imports",
        help="import python file or package. EX: --import 'import users_defined.py' --import 'users.my_func.py'",
    )
    parser.add_argument(
        "--incdir",
        default=[],
        metavar="DIRECTORY",
        action="append",
        type=path_wrapper,
        help="specify a directory to lookup module-level blocks",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="development mode which will keep running and render any template if it has been modified",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="show more informations",
    )
    return parser.parse_args(args)


singleton = {}


def render(
    uri: Path,
    output_directory: Path,
    _globals: dict,
    logger: logging.Logger = None,
    lookup: TemplateLookup = None,
    imports: list = None,
):
    if uri.suffix != ".tpp":
        raise ExtensionError(uri)

    try:
        if str(uri) in singleton:
            tmp = singleton.get(str(uri))
        else:
            tmp = Template(
                # note that must use text instead of filename
                # if lookup template locates at another folder
                uri.open().read(),
                uri=uri.name,
                preprocessor=tpp_preprocessor,
                lookup=lookup,
                imports=imports,
                strict_undefined=False,
                # module_directory=lookup.module_directory,
            )
            singleton[str(uri)] = tmp

        out = output_directory.joinpath(uri.stem)
        msg = f"{'generated'} | {out}"
        with out.open("w") as fo:
            fo.write(tmp.render(store={}, __file__=str(uri), **_globals))
        logger.info(msg) if logger else print(msg)
    except:
        sep = "\n"
        errors = ["  "]
        traceback = exceptions.RichTraceback()
        for (filename, lineno, function, line) in traceback.traceback:
            errors.append("File %s, line %s, in %s" % (filename, lineno, function))
            errors.append(f"  {line}")
        errors.append(
            "%s: %s" % (str(traceback.error.__class__.__name__), traceback.error)
        )
        msg = sep.join(errors)
        logger.error(msg) if logger else print(msg)


def app():
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

    sources = argv.filelist + argv.pathnames
    if not sources:
        raise ValueError("User must specify at least one template source")

    ### setup global variables for template engine
    imports = ImportedModulesMonitor("\n".join(argv.imports))
    _globals = dict(getmembers(filters, isfunction))
    _globals.update(imports.modules)
    _globals["VerilogModule"] = VerilogModule
    _globals["TIE"] = TIE
    _globals["ASSIGN"] = ASSIGN
    _globals["tie0"] = tie0
    _globals["tie_max"] = tie_max
    _globals["defines"] = dict(argv.defines)
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

    ### render templates
    lookup = None
    with tempfile.TemporaryDirectory(prefix=".tmpdir_", dir=Path.cwd()) as tmpdir:
        if argv.incdir:
            lookup = TemplateLookup(
                directories=list(map(str, argv.incdir)),
                preprocessor=tpp_preprocessor,
                module_directory=tmpdir,
                strict_undefined=False,
            )

        def render_all():
            for filename in iter_collect_files(*sources):
                render(
                    filename,
                    output_directory,
                    _globals,
                    lookup=lookup,
                    logger=root_logger,
                )

        render_all()  # always render all once
        if not argv.dev:
            return
        filemonitor = FileMonitor(*sources)
        lookup_monitor = FileMonitor(*argv.incdir)
        print('Running at development mode. Use "ctrl+c" to exit.')
        while True:
            try:
                if imports.try_reload_modules():
                    render_all()

                for event, filename in lookup_monitor.iter_diff(verbose=False):
                    root_logger.info(f"{event} | {filename}")
                    if event != REMOVED:
                        render_all()
                        break

                for event, filename in filemonitor.iter_diff(verbose=False):
                    root_logger.info(f"{event} | {filename}")
                    if event == REMOVED:
                        continue
                    render(
                        filename,
                        output_directory,
                        _globals,
                        lookup=lookup,
                        logger=root_logger,
                    )
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nBye :D\n")
                return
            except Exception as e:
                root_logger.exception(
                    f"{e}: Unexpected erros, please contact the author"
                )
                return 1