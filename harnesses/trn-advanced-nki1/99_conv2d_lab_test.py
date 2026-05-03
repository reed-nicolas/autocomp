"""Marker harness for the lab6 nki_conv2d problem.

The actual evaluation is not driven by this file. ``TrnEvalBackend.evaluate_code``
detects ``prob_id=99`` under ``trn-advanced-nki1`` and dispatches to
``autocomp.backend.trn.lab_conv2d_eval.evaluate_lab_conv2d``, which shells
out to ``python tester.py --basic`` with the candidate dropped in as
``conv2d.py``. The lab tester / utils / ref live under ``~/lab6/nki_conv2d``
(override via ``AUTOCOMP_LAB6_DIR``).
"""
