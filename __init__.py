# This package exists for compatibility with previous releases
# of ScientificPython that supported both NumPy and the old
# Numeric package. Please don't use it in new code, use numpy
# directly.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import sharepy
import l2gen
import sensor
import utils
import calibration_main
import atmoscorr_main
import common

# from .l2gen import *
# from .sharepy import *
# from .sensor import *
# from .utils import *


