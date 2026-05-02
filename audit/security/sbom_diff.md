# SBOM / Advisory DB diff
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

Source: `requirements-prod.txt` (pinned production manifest), checked against the GitHub Advisory Database.

| Package | Pinned | Advisory | Affected | Patched |
|---|---|---|---|---|
| torch | 2.1.2 | heap buffer overflow | `<2.2.0` | `2.2.0` |
| torch | 2.1.2 | use-after-free | `<2.2.0` | `2.2.0` |
| torch | 2.1.2 | torch.load weights_only RCE | `<2.6.0` | `2.6.0` |
| qiskit | 0.45.1 | QPY file DoS | `<1.3.0` | `1.3.0` |
| qiskit | 0.45.1 | QPY arbitrary code execution | `<=1.4.1` | `1.4.2` |
| gunicorn | 21.2.0 | HTTP request smuggling | `<22.0.0` | `22.0.0` |
| gunicorn | 21.2.0 | endpoint restriction bypass | `<22.0.0` | `22.0.0` |

Recommendation: bump torch≥2.6.0, qiskit≥1.4.2, gunicorn≥22.0.0; if torch≥2.6 conflicts with qiskit-aer 0.13.1, plan a coordinated bump.
