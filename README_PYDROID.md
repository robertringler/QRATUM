# Running Micro-Q Simulation on Pydroid (Android)

Steps:
1. Install Pydroid 3 from Google Play.
2. In Pydroid: open Pip (menu) and install dependencies:
   - pip install -r requirements.txt
   or
   - pip install flask requests

3. Create the files in Pydroid's editor:
   - server.py (copy contents)
   - benchmark.py (copy contents)
   - requirements.txt

4. Run the server:
   - In Pydroid, open server.py and Run. The Flask server will bind to 0.0.0.0:5000.
   - You should see "Running on http://0.0.0.0:5000" in the console.

5. Open benchmark:
   - In Pydroid, open benchmark.py and Run (in a separate tab).
   - It will execute local requests to http://127.0.0.1:5000 and print latency summaries.

Notes & caveats:
- This is a simulation: the server is a Python stub, not the Rust qd/qapi binaries.
- Mobile CPUs throttle on sustained load — expect higher latencies as the device warms up.
- Pydroid's Python interpreter is not optimized like native Rust binaries; results are best used for relative measurement and device-level profiling.
- For true Rust benchmarks, consider using Termux or cross-compiling Rust for Android and running native binaries.
