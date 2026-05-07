#!/usr/bin/env bash
# P0.4 — Reproducibility driver (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/../.."
out="out/repro"; mkdir -p "$out"

echo "[1/4] Building kernel..."
cargo build -p qratum-os \
  -Zbuild-std=core,compiler_builtins,alloc \
  -Zbuild-std-features=compiler-builtins-mem >/dev/null

echo "[2/4] Building host arbiter..."
host_target="$(rustc -vV | sed -n 's|host: ||p')"
cargo build -p qratum-arbiter --target "$host_target" >/dev/null

echo "[3/4] Running test matrix..."
cargo test -p qratum-arbiter --target "$host_target" --no-fail-fast --release \
  2>&1 | tee "$out/test_log.txt" >/dev/null

echo "[4/4] Computing hashes..."
kernel="target/x86_64-unknown-uefi/debug/qratum.efi"
kbytes=$(stat -c %s "$kernel" 2>/dev/null || stat -f %z "$kernel")
khash=$(sha256sum "$kernel" | awk '{print $1}')
echo "$khash  $kernel  $kbytes" > "$out/final_state.hash"
ahash=$(sha256sum "$out/test_log.txt" | awk '{print $1}')
echo "$ahash" > "$out/audit.hash"
cat > "$out/replay_receipt.json" <<EOF
{
  "toolchain": "$(rustup show active-toolchain | head -n1)",
  "kernel_bytes": $kbytes,
  "kernel_sha256": "$khash",
  "audit_sha256":  "$ahash",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
grep "test result:" "$out/test_log.txt" \
  | jq -R -s '{ test_summary_lines: split("\n") | map(select(length>0)) }' \
  > "$out/performance_report.json" 2>/dev/null \
  || echo '{"test_summary_lines":[]}' > "$out/performance_report.json"
echo "P0.4 reproduce: OK"
echo "  kernel.sha256 = $khash"
echo "  audit.sha256  = $ahash"
