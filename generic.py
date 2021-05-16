from typing import Generator, List, Union
from pathlib import Path
import re
import os
import glob


def expandpath(path: Path) -> Generator[Path, None, None]:
    path = path.expanduser().resolve()
    parts = path.parts[1:]
    yield from Path(path.root).glob(*parts)


def iter_filelist(filelist: Path) -> Generator[Path, None, None]:
    with filelist.open("r") as fo:
        for line in fo.readlines():
            line = re.sub(r"//.*", "", line).strip()  # remove comment
            if not line:
                continue
            line = os.path.expandvars(line)
            option = None
            for token in re.split("\s+", line):
                if option:
                    # previous token is option, so current token is option value
                    if option in ("-f", "-F"):
                        # a filelist inside a filelist
                        yield from iter_filelist(Path(token))
                    else:
                        yield Path(token)
                    option = None

                elif token[0] == "-":
                    # current token is option
                    option = token
                else:
                    yield Path(token)


def iter_collect_files(*args: Union[Path, None]) -> Generator[Path, None, None]:
    for path in args:
        if not path:
            continue
        if glob.has_magic(str(path)):
            yield from expandpath(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_dir():
            yield from path.iterdir()
        elif path.suffix == ".f":
            yield from iter_filelist(path)
        else:
            yield path


def collect_files(*args: Path, _filter=None) -> list:
    if _filter is None:
        return list(iter_collect_files(*args))
    return list(filter(_filter, iter_collect_files(*args)))
