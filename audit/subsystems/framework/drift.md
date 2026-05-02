# Drift report: `framework`
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:29:11Z

Coexists with the stricter ciir_crs_ric/ package. Both are imported via qagents.__init__.py — risk of two parallel implementations of the same conceptual surface. See GAP-CIIR-MQ-001 for adjacent absence.
