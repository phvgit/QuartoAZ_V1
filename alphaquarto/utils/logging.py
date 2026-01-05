# -*- coding: utf-8 -*-
"""Logging setup"""

def setup_logging(name='alphaquarto'):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
