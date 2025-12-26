## Remaining Code Quality Issues
### Ruff Linting
658	W293  	[ ] blank-line-with-whitespace
232	N806  	[ ] non-lowercase-variable-in-function
198	UP045 	[ ] non-pep604-annotation-optional
192	UP006 	[ ] non-pep585-annotation
185	      	[ ] invalid-syntax
 83	F821  	[ ] undefined-name
 83	F841  	[ ] unused-variable
 50	E402  	[ ] module-import-not-at-top-of-file
 45	N803  	[ ] invalid-argument-name
 45	SIM102	[ ] collapsible-if
 44	B007  	[ ] unused-loop-control-variable
 23	B028  	[ ] no-explicit-stacklevel
 21	F401  	[ ] unused-import
 18	B904  	[ ] raise-without-from-inside-except
 16	C401  	[ ] unnecessary-generator-set
 14	F811  	[ ] redefined-while-unused
 14	W291  	[ ] trailing-whitespace
 13	N802  	[ ] invalid-function-name
 11	SIM108	[ ] if-else-block-instead-of-if-exp
 10	I001  	[*] unsorted-imports
 10	SIM103	[ ] needless-bool
 10	SIM118	[ ] in-dict-keys
  6	N818  	[ ] error-suffix-on-exception-name
  6	N999  	[ ] invalid-module-name
  4	B017  	[ ] assert-raises-exception
  4	SIM105	[ ] suppressible-exception
  3	SIM110	[ ] reimplemented-builtin
  2	B011  	[ ] assert-false
  2	C416  	[ ] unnecessary-comprehension
  2	E722  	[ ] bare-except
  2	E731  	[ ] lambda-assignment
  2	E741  	[ ] ambiguous-variable-name
  2	N812  	[ ] lowercase-imported-as-non-lowercase
  1	C408  	[ ] unnecessary-collection-call
  1	F404  	[ ] late-future-import
  1	N816  	[ ] mixed-case-variable-in-global-scope
  1	SIM113	[ ] enumerate-for-loop
  1	SIM115	[ ] open-file-with-context-handler
  1	SIM201	[ ] negate-equal-op
  1	SIM222	[ ] expr-or-true
  1	SIM401	[ ] if-else-block-instead-of-dict-get
  1	UP007 	[ ] non-pep604-annotation-union
Found 2019 errors.
[*] 10 fixable with the `--fix` option (1271 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
quasim-api is not a valid Python package name
Type checking completed
