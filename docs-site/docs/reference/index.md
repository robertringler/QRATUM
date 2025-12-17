# Reference

Complete reference documentation for QRATUM APIs, CLI, and configuration.

## Documentation Sections

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation with examples
    
    [:octicons-arrow-right-24: API Reference](api-reference.md)

-   :material-console:{ .lg .middle } __CLI Reference__

    ---

    Command-line interface documentation
    
    [:octicons-arrow-right-24: CLI Reference](cli-reference.md)

-   :material-cog:{ .lg .middle } __Configuration__

    ---

    Configuration options and environment variables
    
    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-code-braces:{ .lg .middle } __Code Examples__

    ---

    Copy-paste ready code examples
    
    [:octicons-arrow-right-24: Code Examples](code-examples.md)

</div>

## Quick Reference

### Python API

```python
# Core imports
from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE
from quasim.quantum.qaoa_optimization import QAOA
```

### CLI Commands

```bash
# Hardware calibration
quasim-hcal --detect

# Run validation
python scripts/test_full_stack.py

# Start services
docker-compose up --build
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_LOG_LEVEL` | Log verbosity | `INFO` |
| `JAX_PLATFORM_NAME` | JAX backend | `cpu` |
| `QRATUM_SEED` | Random seed | `None` |
