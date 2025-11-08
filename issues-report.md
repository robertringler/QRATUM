## Remaining Code Quality Issues
### Ruff Linting
136	UP006 	[ ] non-pep585-annotation
115	N806  	[ ] non-lowercase-variable-in-function
 39	W291  	[ ] trailing-whitespace
 36	UP045 	[ ] non-pep604-annotation-optional
 33	      	[ ] invalid-syntax
 29	E402  	[ ] module-import-not-at-top-of-file
 27	I001  	[*] unsorted-imports
 14	F841  	[ ] unused-variable
 13	B007  	[ ] unused-loop-control-variable
  8	F811  	[ ] redefined-while-unused
  7	SIM108	[ ] if-else-block-instead-of-if-exp
  6	F401  	[ ] unused-import
  6	F821  	[ ] undefined-name
  6	N803  	[ ] invalid-argument-name
  6	SIM102	[ ] collapsible-if
  5	SIM103	[ ] needless-bool
  4	N802  	[ ] invalid-function-name
  3	C401  	[ ] unnecessary-generator-set
  3	UP007 	[ ] non-pep604-annotation-union
  1	C416  	[ ] unnecessary-comprehension
  1	F404  	[ ] late-future-import
  1	SIM105	[ ] suppressible-exception
  1	SIM110	[ ] reimplemented-builtin
  1	W293  	[ ] blank-line-with-whitespace
Found 501 errors.
[*] 27 fixable with the `--fix` option (256 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
pyproject.toml: Cannot overwrite a value (at line 43, column 116)
quasim-api is not a valid Python package name
Type checking completed
