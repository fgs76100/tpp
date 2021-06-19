import ast
import importlib
from typing import Callable, Generator, Optional, List, Mapping
from .monitors import MonitorBase
from pathlib import Path
from inspect import getmembers, ismodule


class ImportModulesMonitor(MonitorBase):
    def __init__(self, code: str, *args: Optional[List[Path]], **kwargs: dict) -> None:
        self.tree = ast.parse(code)
        self.declared_identifiers = set()
        self.modules = dict()
        self._walk()
        super().__init__(*args, **kwargs)
        self.sub_modules_before = dict()

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
                for alias in node.names:
                    _id = alias.asname or alias.name
                    self.modules[_id] = importlib.import_module(
                        f"{module}.{alias.name}"
                    )

    def iter_module_filepath(self) -> Generator[Path, None, None]:
        for module in self.modules.values():
            yield Path(module.__file__)

    def get_module_filepath(self, mapper: Callable = None):
        mapper = mapper or (lambda x: x)
        return list(map(mapper, self.iter_module_filepath))

    def reload_modules(self) -> bool:
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
