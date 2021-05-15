from generic import iter_collect_files
from pathlib import Path
from typing import Generator, List, Callable, Mapping, Union
import logging

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

        return True

    def get_status(self) -> Mapping[str, float]:
        raise NotImplementedError

    def iter_diff(self, verbose=True) -> Generator[Path, None, None]:
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
            " | {0} | {1}".format(
                event.upper(),
                item,
            )
        )


class FileMonitor(MonitorBase):
    def filter_target(self, target: Path):
        if not target.suffix == ".tpp":
            return False
        return super().filter_target(target)

    def get_status(self) -> Mapping[str, float]:
        return dict([(str(path), path.stat().st_mtime) for path in self.targets])
