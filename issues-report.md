## Remaining Code Quality Issues
### Ruff Linting
125	UP006 	non-pep585-annotation
115	N806  	non-lowercase-variable-in-function
 39	W291  	trailing-whitespace
 30	      	invalid-syntax
 19	UP045 	non-pep604-annotation-optional
 14	F841  	unused-variable
 13	B007  	unused-loop-control-variable
  8	E402  	module-import-not-at-top-of-file
  7	SIM108	if-else-block-instead-of-if-exp
  6	F821  	undefined-name
  6	N803  	invalid-argument-name
  6	SIM102	collapsible-if
  5	SIM103	needless-bool
  4	N802  	invalid-function-name
  3	C401  	unnecessary-generator-set
  3	F811  	redefined-while-unused
  3	UP007 	non-pep604-annotation-union
  2	F401  	unused-import
  1	C416  	unnecessary-comprehension
  1	F404  	late-future-import
  1	SIM105	suppressible-exception
  1	SIM110	reimplemented-builtin
Found 412 errors.
No fixes available (227 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
pyproject.toml: [mypy]: python_version: Python 3.8 is not supported (must be 3.9 or higher)
quasim-api is not a valid Python package name
Type checking completed
