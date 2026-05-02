## Remaining Code Quality Issues
### Ruff Linting
677	W293  	[ ] blank-line-with-whitespace
535	N806  	[ ] non-lowercase-variable-in-function
389	UP045 	[ ] non-pep604-annotation-optional
331	N999  	[ ] invalid-module-name
296	UP006 	[ ] non-pep585-annotation
145	F841  	[ ] unused-variable
125	F821  	[ ] undefined-name
 96	E402  	[ ] module-import-not-at-top-of-file
 92	N803  	[ ] invalid-argument-name
 85	SIM102	[ ] collapsible-if
 76	B007  	[ ] unused-loop-control-variable
 44	F811  	[ ] redefined-while-unused
 39	F401  	[ ] unused-import
 33	W291  	[ ] trailing-whitespace
 31	N802  	[ ] invalid-function-name
 29	SIM108	[ ] if-else-block-instead-of-if-exp
 28	B904  	[ ] raise-without-from-inside-except
 24	C401  	[ ] unnecessary-generator-set
 23	B028  	[ ] no-explicit-stacklevel
 23	SIM103	[ ] needless-bool
 22	SIM105	[ ] suppressible-exception
 16	SIM118	[ ] in-dict-keys
 15	E741  	[ ] ambiguous-variable-name
 14	B023  	[ ] function-uses-loop-variable
 14	I001  	[*] unsorted-imports
 10	N818  	[ ] error-suffix-on-exception-name
 10	UP031 	[ ] printf-string-formatting
  9	E722  	[ ] bare-except
  9	SIM110	[ ] reimplemented-builtin
  8	N815  	[ ] mixed-case-variable-in-class-scope
  6	C416  	[ ] unnecessary-comprehension
  5	B017  	[ ] assert-raises-exception
  4	SIM115	[ ] open-file-with-context-handler
  3	C408  	[ ] unnecessary-collection-call
  3	N801  	[ ] invalid-class-name
  2	B011  	[ ] assert-false
  2	E731  	[ ] lambda-assignment
  2	N812  	[ ] lowercase-imported-as-non-lowercase
  2	SIM113	[ ] enumerate-for-loop
  1	      	[ ] invalid-syntax
  1	F402  	[ ] import-shadowed-by-loop-var
  1	F404  	[ ] late-future-import
  1	F509  	[ ] percent-format-unsupported-format-character
  1	F601  	[ ] multi-value-repeated-key-literal
  1	N811  	[ ] constant-imported-as-non-constant
  1	N816  	[ ] mixed-case-variable-in-global-scope
  1	SIM101	[ ] duplicate-isinstance-call
  1	SIM201	[ ] negate-equal-op
  1	SIM222	[ ] expr-or-true
  1	SIM401	[ ] if-else-block-instead-of-dict-get
  1	UP007 	[ ] non-pep604-annotation-union
Found 3289 errors.
[*] 14 fixable with the `--fix` option (1785 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
quasim-api contains __init__.py but is not a valid Python package name
Type checking completed
