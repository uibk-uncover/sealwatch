
import logging
import os
import sealwatch as sw
import tempfile
import unittest


class TestBackend(unittest.TestCase):
    """Test suite for backend."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    def test_backend(self):
        self._logger.info('TestBackend.test_backend()')
        self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
        sw.set_backend(sw.BACKEND_RUST)
        self.assertIs(sw.get_backend(), sw.BACKEND_RUST)
        sw.set_backend(sw.BACKEND_PYTHON)
        self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)

    def test_with(self):
        self._logger.info('TestBackend.test_with()')
        self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
        # switch to rust
        with sw.BACKEND_RUST:
            self.assertIs(sw.get_backend(), sw.BACKEND_RUST)
            # switch to python
            with sw.BACKEND_PYTHON:
                self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
            # back to rust
            self.assertIs(sw.get_backend(), sw.BACKEND_RUST)
        # back to default (python)
        self.assertIs(sw.get_backend(), sw.BACKEND_PYTHON)
