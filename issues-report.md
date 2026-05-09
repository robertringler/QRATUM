## Remaining Code Quality Issues
### Ruff Linting
1356	N806  	[ ] non-lowercase-variable-in-function
 679	W293  	[ ] blank-line-with-whitespace
 523	UP006 	[ ] non-pep585-annotation
 455	UP045 	[ ] non-pep604-annotation-optional
 345	N999  	[ ] invalid-module-name
 204	F841  	[ ] unused-variable
 199	N803  	[ ] invalid-argument-name
 169	      	[ ] invalid-syntax
 127	F821  	[ ] undefined-name
  97	E402  	[ ] module-import-not-at-top-of-file
  86	SIM102	[ ] collapsible-if
  84	B007  	[ ] unused-loop-control-variable
  56	N802  	[ ] invalid-function-name
  47	C408  	[ ] unnecessary-collection-call
  42	F811  	[ ] redefined-while-unused
  40	E741  	[ ] ambiguous-variable-name
  35	F401  	[ ] unused-import
  34	W291  	[ ] trailing-whitespace
  31	SIM105	[ ] suppressible-exception
  30	SIM108	[ ] if-else-block-instead-of-if-exp
  28	B904  	[ ] raise-without-from-inside-except
  24	C401  	[ ] unnecessary-generator-set
  23	B028  	[ ] no-explicit-stacklevel
  23	SIM103	[ ] needless-bool
  18	B023  	[ ] function-uses-loop-variable
  16	SIM118	[ ] in-dict-keys
  13	I001  	[*] unsorted-imports
  12	N815  	[ ] mixed-case-variable-in-class-scope
  11	N818  	[ ] error-suffix-on-exception-name
  10	SIM110	[ ] reimplemented-builtin
  10	UP031 	[ ] printf-string-formatting
   8	E722  	[ ] bare-except
   6	C416  	[ ] unnecessary-comprehension
   6	E731  	[ ] lambda-assignment
   6	SIM115	[ ] open-file-with-context-handler
   5	B017  	[ ] assert-raises-exception
   4	N801  	[ ] invalid-class-name
   3	SIM113	[ ] enumerate-for-loop
   2	B011  	[ ] assert-false
   2	N812  	[ ] lowercase-imported-as-non-lowercase
   1	F402  	[ ] import-shadowed-by-loop-var
   1	F404  	[ ] late-future-import
   1	F509  	[ ] percent-format-unsupported-format-character
   1	F601  	[ ] multi-value-repeated-key-literal
   1	N816  	[ ] mixed-case-variable-in-global-scope
   1	SIM101	[ ] duplicate-isinstance-call
   1	SIM201	[ ] negate-equal-op
   1	SIM222	[ ] expr-or-true
   1	SIM401	[ ] if-else-block-instead-of-dict-get
   1	UP007 	[ ] non-pep604-annotation-union
Found 4879 errors.
[*] 13 fixable with the `--fix` option (2208 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
quasim-api contains __init__.py but is not a valid Python package name
Type checking completed
