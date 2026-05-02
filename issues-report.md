## Remaining Code Quality Issues
### Ruff Linting
1957	W293  	[ ] blank-line-with-whitespace
1353	N806  	[ ] non-lowercase-variable-in-function
 772	I001  	[*] unsorted-imports
 427	UP045 	[ ] non-pep604-annotation-optional
 410	UP006 	[ ] non-pep585-annotation
 340	N999  	[ ] invalid-module-name
 204	F841  	[ ] unused-variable
 199	N803  	[ ] invalid-argument-name
 169	      	[ ] invalid-syntax
 127	F821  	[ ] undefined-name
  94	E402  	[ ] module-import-not-at-top-of-file
  85	SIM102	[ ] collapsible-if
  84	B007  	[ ] unused-loop-control-variable
  56	N802  	[ ] invalid-function-name
  47	C408  	[ ] unnecessary-collection-call
  43	F811  	[ ] redefined-while-unused
  42	W291  	[ ] trailing-whitespace
  40	E741  	[ ] ambiguous-variable-name
  35	F401  	[ ] unused-import
  30	SIM108	[ ] if-else-block-instead-of-if-exp
  28	B904  	[ ] raise-without-from-inside-except
  24	C401  	[ ] unnecessary-generator-set
  23	B028  	[ ] no-explicit-stacklevel
  23	SIM103	[ ] needless-bool
  22	SIM105	[ ] suppressible-exception
  18	B023  	[ ] function-uses-loop-variable
  16	SIM118	[ ] in-dict-keys
  12	N815  	[ ] mixed-case-variable-in-class-scope
  10	N818  	[ ] error-suffix-on-exception-name
  10	SIM110	[ ] reimplemented-builtin
  10	UP031 	[ ] printf-string-formatting
   9	E722  	[ ] bare-except
   6	C416  	[ ] unnecessary-comprehension
   6	E731  	[ ] lambda-assignment
   5	B017  	[ ] assert-raises-exception
   4	E701  	[ ] multiple-statements-on-one-line-colon
   4	N801  	[ ] invalid-class-name
   4	SIM115	[ ] open-file-with-context-handler
   3	SIM113	[ ] enumerate-for-loop
   2	B011  	[ ] assert-false
   2	E702  	[ ] multiple-statements-on-one-line-semicolon
   2	N812  	[ ] lowercase-imported-as-non-lowercase
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
Found 6768 errors.
[*] 772 fixable with the `--fix` option (3342 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
pyproject.toml: Cannot overwrite a value (at line 103, column 36)
quasim-api contains __init__.py but is not a valid Python package name
Type checking completed
