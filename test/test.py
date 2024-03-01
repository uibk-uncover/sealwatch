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
# from test_crm import *
# from test_jrm import *
# from test_fld_ensemble import *
# from test_grouping import *
# from test_pharm import *
# from test_spam import *
# ==================

# run unittests
if __name__ == "__main__":
    unittest.main()
