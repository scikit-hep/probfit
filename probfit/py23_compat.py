"""
Python 2 / 3 compatibility helpers.
"""
import sys

py_ver = sys.version_info
PY2 = False
PY3 = False
if py_ver[0] == 2:
    PY2 = True
else:  # just in case PY4
    PY3 = True


if PY2:
    range = xrange
else:
    range = range
