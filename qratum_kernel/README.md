# qratum_kernel — headless boot path

The QRATUM kernel daemon is the OS-style entrypoint that lets QRATUM boot
without the Unreal Engine simulation. It implements the same Φ=2
deterministic kernels as the UE5 [`UQRATUMBridgeSubsystem`](../soi/unreal_bridge/Source/QRATUM/Public/Bridge/QRATUMBridgeSubsystem.h) and publishes the same JSON state contract.

## Boot

```cmd
:: Windows
qratum-boot.cmd               :: kernel + desktop shell
qratum-boot.cmd --with-ue5    :: + UE5 simulation
qratum-boot.cmd --halt
```

```bash
# Linux / macOS
./qratum-boot.sh boot
./qratum-boot.sh halt
./qratum-boot.sh status
```

## State contract

Atomic JSON snapshot at:

| Platform | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\QRATUM\bridge\state.json` |
| Linux   | `$XDG_DATA_HOME/QRATUM/bridge/state.json` (or `~/.local/share/...`) |
| macOS   | `~/Library/Application Support/QRATUM/bridge/state.json` |

Schema fields: `phase`, `epoch_tick`, `ciir.{phi,delta_w_norm}`,
`crs.{active_stratum,iss_residual}`, `ric_dispatch_weights[]`,
`quasim.{s,quantum_bound}`, `qubic.{epoch,nodes}`, `rmhd.{reconnection_rate,psi_flux}`,
`mvri.{ema,gate_mask,status}`, `rwb.sequence`, `cgl.{lambda_min,rank_deficient}`.

Both publishers (Python daemon + UE5 subsystem) write to the same path with
atomic temp+rename semantics, so the QRATUM Desktop shell can attach to
either source transparently.

## Manual run

```bash
python qratum_kernel/qratum_kernel_daemon.py --hz 60
python qratum_kernel/qratum_kernel_daemon.py --steps 1200 --quiet  # smoke
```
