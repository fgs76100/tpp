from .generic import iter_collect_files
from pathlib import Path
from typing import Generator, List, Callable, Mapping, Tuple, Union, Optional
import logging
import ast
import importlib
from inspect import getmembers, ismodule

# from generic import get_now, iglob, iter_filelist_reader
# from event import EventManger, ANY_EVENT

MODIFIED = "modified"
REMOVED = "removed"
ADDED = "added"


class MonitorBase:
    def __init__(self, *args: Union[List[Path], None], **kwargs: dict):

        # if not isinstance(targets, list):
        #     raise TypeError("input should be a list")

        self.logger = logging.getLogger(__name__)
        self._targets = args
        self.ignores = kwargs.get("ignores", None) or []

        # self.before = self.get_status()
        self.before = {}
        self.initialize()

    def iter_targets(self, _filter: Callable = None) -> Generator[Path, None, None]:
        _filter = _filter or self.filter_target
        for target in iter_collect_files(*self._targets):
            if _filter(target):
                yield target

    @property
    def targets(self) -> List[Path]:
        return list(self.iter_targets())

    def initialize(self):
        self.before = self.get_status()

    def filter_target(self, target: Path) -> bool:
        for ignore in self.ignores:
            if target.match(ignore):
                self.logger.debug(f"user ignores: {target}")
                return False

        if not target.exists():
            self.logger.error(f"No such file or directory: {target}")
            return False

        if target.name[0] == ".":
            # ignore hidden files
            return False

        return True

    def get_status(self) -> Mapping[str, float]:
        raise NotImplementedError

    def iter_diff(self, verbose=True) -> Generator[Tuple[str, Path], None, None]:
        before = self.before
        after = self.get_status()
        events = {
            ADDED: [f for f in after if f not in before],
            REMOVED: [f for f in before if f not in after],
        }
        events[MODIFIED] = [
            f
            for f, mtime in after.items()
            if f not in events[ADDED] and mtime != before[f]
        ]
        self.before = after
        for event, items in events.items():
            for item in items:
                if verbose:
                    self.verbose(event, item, before.get(item, -1), after.get(item, -1))
                yield event, Path(item)

    def diff(self, verbose=True) -> List[Path]:
        return list(self.iter_diff(verbose=verbose))

    def verbose(self, event, item, before, after):
        self.logger.info(
            " {0} | {1}".format(
                event.upper(),
                item,
            )
        )


class FileMonitor(MonitorBase):
    def get_status(self) -> Mapping[str, float]:
        return dict([(str(path), path.stat().st_mtime) for path in self.targets])


class ImportedModulesMonitor(MonitorBase):
    def __init__(
        self, code: str, package=None, *args: Optional[List[Path]], **kwargs: dict
    ) -> None:
        self.tree = ast.parse(code)
        self.declared_identifiers = set()
        self.modules = dict()
        self.package = package
        self._walk()
        super().__init__(*args, **kwargs)

    def _walk(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    _id = alias.asname or module.rsplit(".", 1)[-1]
                    module = importlib.import_module(f"{module}")
                    self.modules[_id] = module
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                relative_import = node.level
                for alias in node.names:
                    _id = alias.asname or alias.name
                    if not relative_import:
                        self.modules[_id] = importlib.import_module(
                            f"{module}.{alias.name}"
                        )
                    else:
                        print(module, _id, __name__, Path.cwd(), self.package)
                        print(ast.dump(node))
                        self.modules[_id] = importlib.import_module(
                            f".{module}.{alias.name}", package=self.package
                        )

    def iter_module_filepath(self) -> Generator[Path, None, None]:
        for module in self.modules.values():
            yield Path(module.__file__)

    def get_module_filepath(self, mapper: Callable = None):
        mapper = mapper or (lambda x: x)
        return list(map(mapper, self.iter_module_filepath))

    def try_reload_modules(self) -> bool:
        """
        return True if any modules had been reloaded
        """
        is_reload = False
        for _, item in self.iter_diff():
            module_id = str(item).split("/", 1)[0]
            module = self.modules[module_id]
            self.modules[module_id] = importlib.reload(module)
            is_reload = True
        return is_reload

    def get_status(self) -> Mapping[str, float]:
        status = []
        for id, module in self.modules.items():
            path = Path(module.__file__)
            status.append((id, path.stat().st_mtime))
            for name, submodule in getmembers(module, ismodule):
                if hasattr(submodule, "__file__"):
                    status.append(
                        (f"{id}/{name}", Path(submodule.__file__).stat().st_mtime)
                    )
        return dict(status)

    def verbose(self, event, item, before, after):
        self.logger.info(f"{event} | module: {item}")

    # def iter_diff(self, verbose: bool = True) -> Generator[Tuple[str, str], None, None]:
    #     for event, item in super().iter_diff(verbose=verbose):
    #         item = str(item).split("/", 1)[0]
    #         yield event, str(item)
