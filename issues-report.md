## Remaining Code Quality Issues
### Ruff Linting
1694	      	[ ] invalid-syntax
 953	W293  	[ ] blank-line-with-whitespace
 501	N806  	[ ] non-lowercase-variable-in-function
 384	UP045 	[ ] non-pep604-annotation-optional
 331	N999  	[ ] invalid-module-name
 283	UP006 	[ ] non-pep585-annotation
 144	F841  	[ ] unused-variable
 125	F821  	[ ] undefined-name
  92	N803  	[ ] invalid-argument-name
  89	E402  	[ ] module-import-not-at-top-of-file
  76	SIM102	[ ] collapsible-if
  67	B007  	[ ] unused-loop-control-variable
  38	F401  	[ ] unused-import
  33	W291  	[ ] trailing-whitespace
  31	N802  	[ ] invalid-function-name
  28	B904  	[ ] raise-without-from-inside-except
  28	F811  	[ ] redefined-while-unused
  28	SIM108	[ ] if-else-block-instead-of-if-exp
  23	B028  	[ ] no-explicit-stacklevel
  21	SIM103	[ ] needless-bool
  20	C401  	[ ] unnecessary-generator-set
  20	SIM105	[ ] suppressible-exception
  14	B023  	[ ] function-uses-loop-variable
  14	E741  	[ ] ambiguous-variable-name
  13	I001  	[*] unsorted-imports
  10	SIM118	[ ] in-dict-keys
  10	UP031 	[ ] printf-string-formatting
   9	E722  	[ ] bare-except
   8	N815  	[ ] mixed-case-variable-in-class-scope
   8	SIM110	[ ] reimplemented-builtin
   7	N818  	[ ] error-suffix-on-exception-name
   5	B017  	[ ] assert-raises-exception
   4	C416  	[ ] unnecessary-comprehension
   4	SIM115	[ ] open-file-with-context-handler
   3	C408  	[ ] unnecessary-collection-call
   3	N801  	[ ] invalid-class-name
   2	B011  	[ ] assert-false
   2	E702  	[ ] multiple-statements-on-one-line-semicolon
   2	E731  	[ ] lambda-assignment
   2	N812  	[ ] lowercase-imported-as-non-lowercase
   2	SIM113	[ ] enumerate-for-loop
   1	E701  	[ ] multiple-statements-on-one-line-colon
   1	F404  	[ ] late-future-import
   1	F509  	[ ] percent-format-unsupported-format-character
   1	N811  	[ ] constant-imported-as-non-constant
   1	N816  	[ ] mixed-case-variable-in-global-scope
   1	SIM101	[ ] duplicate-isinstance-call
   1	SIM201	[ ] negate-equal-op
   1	SIM222	[ ] expr-or-true
   1	SIM401	[ ] if-else-block-instead-of-dict-get
   1	UP007 	[ ] non-pep604-annotation-union
Found 5141 errors.
[*] 13 fixable with the `--fix` option (1738 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
quasim-api contains __init__.py but is not a valid Python package name
Type checking completed
