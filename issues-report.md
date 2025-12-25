## Remaining Code Quality Issues
### Ruff Linting
1020	W293  	[ ] blank-line-with-whitespace
 265	I001  	[*] unsorted-imports
 232	N806  	[ ] non-lowercase-variable-in-function
 197	UP045 	[ ] non-pep604-annotation-optional
 185	      	[ ] invalid-syntax
 179	UP006 	[ ] non-pep585-annotation
  83	F821  	[ ] undefined-name
  74	F841  	[ ] unused-variable
  50	E402  	[ ] module-import-not-at-top-of-file
  45	N803  	[ ] invalid-argument-name
  40	B007  	[ ] unused-loop-control-variable
  31	SIM102	[ ] collapsible-if
  23	B028  	[ ] no-explicit-stacklevel
  21	F401  	[ ] unused-import
  19	W291  	[ ] trailing-whitespace
  18	B904  	[ ] raise-without-from-inside-except
  14	C401  	[ ] unnecessary-generator-set
  14	F811  	[ ] redefined-while-unused
  13	N802  	[ ] invalid-function-name
  10	SIM108	[ ] if-else-block-instead-of-if-exp
  10	SIM118	[ ] in-dict-keys
   7	E722  	[ ] bare-except
   6	N999  	[ ] invalid-module-name
   4	N818  	[ ] error-suffix-on-exception-name
   4	SIM103	[ ] needless-bool
   4	SIM105	[ ] suppressible-exception
   3	SIM110	[ ] reimplemented-builtin
   2	B011  	[ ] assert-false
   2	B017  	[ ] assert-raises-exception
   2	E731  	[ ] lambda-assignment
   2	E741  	[ ] ambiguous-variable-name
   2	N812  	[ ] lowercase-imported-as-non-lowercase
   1	C408  	[ ] unnecessary-collection-call
   1	C416  	[ ] unnecessary-comprehension
   1	F404  	[ ] late-future-import
   1	N801  	[ ] invalid-class-name
   1	N816  	[ ] mixed-case-variable-in-global-scope
   1	SIM115	[ ] open-file-with-context-handler
   1	SIM201	[ ] negate-equal-op
   1	SIM401	[ ] if-else-block-instead-of-dict-get
   1	UP007 	[ ] non-pep604-annotation-union
Found 2590 errors.
[*] 265 fixable with the `--fix` option (1595 hidden fixes can be enabled with the `--unsafe-fixes` option).
### Type Checking (mypy)
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/opt/hostedtoolcache/Python/3.12.12/x64/lib/python3.12/site-packages/mypy/__main__.py", line 9, in <module>
    from mypy.main import main, process_options
  File "mypy/main.py", line 43, in <module>
AttributeError: module 'platform' has no attribute 'python_implementation'
Type checking completed
