from __future__ import absolute_import, division, print_function

import logging.config
import os
import yaml

from . import data

from data import *

__all__ = ['data']
__all__.extend(data.__all__)

__logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
if os.path.exists(__logging_config_path):
    with open(__logging_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.getLogger('').addHandler(logging.NullHandler())
