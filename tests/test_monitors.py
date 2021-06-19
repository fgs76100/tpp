import unittest
from tpp.monitors import FileMonitor, ImportedModulesMonitor
import tempfile
import random
import pathlib
import time
import os
from typing import List


@unittest.skip
class TestFileMonitor(unittest.TestCase):
    def test_file_monitor(self):
        with tempfile.TemporaryDirectory(dir="./") as tmpdir:
            file_cnts = random.randint(10, 20)
            files: List[pathlib.Path] = []
            for _ in range(file_cnts):
                tmpfile = tempfile.NamedTemporaryFile("w", dir=tmpdir, delete=False)
                files.append(pathlib.Path(tmpfile.name))
                tmpfile.close()

            # fm = FileMonitor(*files)
            fm = FileMonitor(pathlib.Path(tmpdir))
            self.assertEqual(len(fm.targets), file_cnts)

            for _ in range(10):
                answer = []
                tmpfiles = random.choices(files, k=file_cnts // 2)
                for tmpfile in tmpfiles:
                    # print(tmpfile)
                    f = tmpfile.open("w", buffering=1)
                    try:
                        f.write("123")
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        f.close()
                    if tmpfile.name not in answer:
                        answer.append(tmpfile.name)

                time.sleep(1)
                diff = fm.diff()
                # print(diff)
                self.assertEqual(len(diff), len(answer))


class TestImportedModuleMonitor(unittest.TestCase):
    def test_imported_module_monitor(self):
        # im = ImportedModulesMonitor("import test_top")
        im = ImportedModulesMonitor("import os")
        im = ImportedModulesMonitor("from os import path")
        im = ImportedModulesMonitor("import os.path as path")
        im = ImportedModulesMonitor("from tpp import generic")
        im = ImportedModulesMonitor("from tpp.vpp import core")
        # im = ImportedModulesMonitor(
        #     "from .test_vpp import test_files_dir", package=__name__
        # )
