## Remaining Code Quality Issues
### Ruff Linting
210	N806  	[ ] non-lowercase-variable-in-function
151	I001  	[*] unsorted-imports
138	W293  	[ ] blank-line-with-whitespace
107	UP006 	[ ] non-pep585-annotation
 49	E402  	[ ] module-import-not-at-top-of-file
 36	N803  	[ ] invalid-argument-name
 24	UP045 	[ ] non-pep604-annotation-optional
 15	F811  	[ ] redefined-while-unused
 14	      	[ ] invalid-syntax
 13	N802  	[ ] invalid-function-name
 11	F401  	[ ] unused-import
 11	SIM102	[ ] collapsible-if
  9	B904  	[ ] raise-without-from-inside-except
  8	B007  	[ ] unused-loop-control-variable
  8	F841  	[ ] unused-variable
  7	C401  	[ ] unnecessary-generator-set
  7	F821  	[ ] undefined-name
  6	N999  	[ ] invalid-module-name
  4	SIM108	[ ] if-else-block-instead-of-if-exp
  2	B011  	[ ] assert-false
  2	E722  	[ ] bare-except
  2	F404  	[ ] late-future-import
  2	N818  	[ ] error-suffix-on-exception-name
  2	SIM103	[ ] needless-bool
  1	E741  	[ ] ambiguous-variable-name
  1	N816  	[ ] mixed-case-variable-in-global-scope
  1	SIM105	[ ] suppressible-exception
  1	SIM118	[ ] in-dict-keys
  1	SIM401	[ ] if-else-block-instead-of-dict-get
Found 843 errors.
[*] 151 fixable with the `--fix` option (291 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
pyproject.toml: Cannot declare ('tool', 'setuptools', 'packages', 'find') twice (at line 99, column 31)
quasim-api is not a valid Python package name
Type checking completed
