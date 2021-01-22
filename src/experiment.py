"""Experiments infrastructure.

This module contains functions with preparations for an experiment.
"""

import os

from config import cfg


def init():
    r"""
    Checks, if results folder has already been used and prevents overwriting of the results.
    Returns:
        None
    """
    old, new = cfg.RESULTS_ROOT / 'cfg.py', cfg.ROOT_DIR / 'config' / 'cfg.py'
    if old.exists():
        if os.system(f'cmp --silent {old} {new}'):
            raise EnvironmentError('Config file in RESULTS_ROOT already exists and differs from the one in config/')
    os.system(f'cp {new} {old}')
