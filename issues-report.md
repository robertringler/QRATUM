## Remaining Code Quality Issues
### Ruff Linting
125	UP006 	non-pep585-annotation
115	N806  	non-lowercase-variable-in-function
 39	W291  	trailing-whitespace
 35	      	invalid-syntax
 19	UP045 	non-pep604-annotation-optional
 14	F841  	unused-variable
 13	B007  	unused-loop-control-variable
 13	W293  	blank-line-with-whitespace
  7	SIM108	if-else-block-instead-of-if-exp
  6	F821  	undefined-name
  6	N803  	invalid-argument-name
  4	N802  	invalid-function-name
  4	SIM102	collapsible-if
  3	C401  	unnecessary-generator-set
  3	E402  	module-import-not-at-top-of-file
  3	SIM103	needless-bool
  3	SIM105	suppressible-exception
  3	UP007 	non-pep604-annotation-union
  2	E722  	bare-except
  2	F401  	unused-import
  2	F811  	redefined-while-unused
  1	C416  	unnecessary-comprehension
  1	F404  	late-future-import
  1	N818  	error-suffix-on-exception-name
  1	SIM110	reimplemented-builtin
  1	SIM115	open-file-with-context-handler
Found 426 errors.
No fixes available (240 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
pyproject.toml: [mypy]: python_version: Python 3.8 is not supported (must be 3.9 or higher)
quasim-api is not a valid Python package name
Type checking completed
