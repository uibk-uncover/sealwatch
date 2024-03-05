import sys
import unittest
sys.path.append(".")


# logging
if __name__ == "__main__":
    import logging
    logging.basicConfig(filename="test.log", level=logging.INFO)
    import jpeglib
    logging.info(f"{jpeglib.__path__=}")


# === unit tests ===
from test_dctr import *
from test_jrm import *
from test_gfr import *
from test_grouping import *
from test_pharm import *
from test_spam import *
from test_fld_ensemble_classifier import *
# ==================

# run unittests
if __name__ == "__main__":
    unittest.main()
