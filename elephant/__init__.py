# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from . import statistics
from . import conversion
from elephant import spikecorr
from . import surrogates
from . import neo_tools


try:
    from . import pandas_bridge
except ImportError:
    pass
