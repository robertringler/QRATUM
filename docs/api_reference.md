# API Reference

## C++ `libquasim`

```cpp
struct QuasimTensor {
    std::vector<int64_t> shape;
    std::vector<std::complex<double>> data;
};

class QuasimRuntime {
public:
    explicit QuasimRuntime(const QuasimConfig &config);
    QuasimTensor contract(const QuasimCircuit &circuit);
    void submit_tensor_job(const TensorJob &job);
};
```

* `QuasimRuntime::contract` executes a tensor-network circuit using the GPU scheduler.
* `submit_tensor_job` enqueues asynchronous work that maps directly to GPU command buffers.

## Python Module `quasim`

```python
import quasim

cfg = quasim.Config(simulation_precision="fp8")
with quasim.runtime(cfg) as rt:
    result = rt.simulate(circuit)
```

* `quasim.runtime` is a context manager that configures the shared memory pool via the `gb10_mm` driver bindings.
* `simulate` dispatches operations through the `libquasim` C++ extension and returns numpy arrays for further analysis.
