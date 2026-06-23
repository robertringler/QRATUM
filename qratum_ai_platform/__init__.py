"""QRATUM AI Platform - remaining auxiliary services.

The platform's importable packages (core, bioinformatics, compliance,
observability, opt, quantum, workflows, algorithms, config) have moved to
their intended home in the ``qratum`` package — their internal absolute
imports were always written as ``qratum.*``. Import them via ``qratum.core``,
``qratum.bioinformatics``, etc. This directory retains only auxiliary assets
(services, pipelines, mlflow, infra, ci, docs) and their tests.
"""

__version__ = "1.0.0"
__all__ = ["services", "tests"]
