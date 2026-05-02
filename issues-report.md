## Remaining Code Quality Issues
### Ruff Linting
1321	      	[ ] invalid-syntax
 957	W293  	[ ] blank-line-with-whitespace
 294	N806  	[ ] non-lowercase-variable-in-function
 289	UP045 	[ ] non-pep604-annotation-optional
 195	UP006 	[ ] non-pep585-annotation
 125	F821  	[ ] undefined-name
 125	F841  	[ ] unused-variable
  84	E402  	[ ] module-import-not-at-top-of-file
  77	SIM102	[ ] collapsible-if
  67	B007  	[ ] unused-loop-control-variable
  56	N803  	[ ] invalid-argument-name
  38	F401  	[ ] unused-import
  29	W291  	[ ] trailing-whitespace
  28	B904  	[ ] raise-without-from-inside-except
  23	B028  	[ ] no-explicit-stacklevel
  22	SIM108	[ ] if-else-block-instead-of-if-exp
  21	C401  	[ ] unnecessary-generator-set
  17	SIM103	[ ] needless-bool
  14	F811  	[ ] redefined-while-unused
  14	SIM105	[ ] suppressible-exception
  14	SIM118	[ ] in-dict-keys
  13	N802  	[ ] invalid-function-name
  10	I001  	[*] unsorted-imports
   6	N818  	[ ] error-suffix-on-exception-name
   6	N999  	[ ] invalid-module-name
   6	SIM110	[ ] reimplemented-builtin
   4	B017  	[ ] assert-raises-exception
   4	C416  	[ ] unnecessary-comprehension
   2	B011  	[ ] assert-false
   2	E722  	[ ] bare-except
   2	E731  	[ ] lambda-assignment
   2	E741  	[ ] ambiguous-variable-name
   2	N812  	[ ] lowercase-imported-as-non-lowercase
   2	SIM113	[ ] enumerate-for-loop
   1	C408  	[ ] unnecessary-collection-call
   1	F404  	[ ] late-future-import
   1	N816  	[ ] mixed-case-variable-in-global-scope
   1	SIM101	[ ] duplicate-isinstance-call
   1	SIM115	[ ] open-file-with-context-handler
   1	SIM201	[ ] negate-equal-op
   1	SIM222	[ ] expr-or-true
   1	SIM401	[ ] if-else-block-instead-of-dict-get
   1	UP007 	[ ] non-pep604-annotation-union
Found 3880 errors.
[*] 10 fixable with the `--fix` option (1513 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
quasim-api contains __init__.py but is not a valid Python package name
Type checking completed
