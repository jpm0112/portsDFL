"""
Package marker for the `models` subpackage.

A file named ``__init__.py`` tells Python that the folder containing it is a
"package" — an importable group of modules. That is what makes imports like
``from .models.registry import build`` work elsewhere in this project.

This file is intentionally (almost) empty: each model lives in its own module
(``bhm_baseline.py``, ``bhm_covariates.py``, ...) and the lookup table that
maps a model key to its builder lives in ``registry.py``. Keeping this file
free of imports avoids import-order surprises (e.g. importing the package
would otherwise eagerly pull in PyMC). Import what you need directly, e.g.::

    from .models.registry import build, set_predict_data
"""
