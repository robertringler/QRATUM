# QRATUM Persistent Audit — Format v2 (P0.3)

> Hybrid log: lock-free in-memory MPSC ring (P0.2) + append-only,
> hash-chained on-disk segment file. Replay reconstructs the kernel's
> arbiter state byte-for-byte.

## 1. Goals

* Crash-safe — torn writes must be detectable and truncatable.
* Hash-chained — every frame depends on its predecessor's hash.
* DMA-ready — writes batched into 4 KiB-aligned segments.
* Lossless — every audit event the kernel emitted before a crash is
  recoverable up to the last fully-committed segment.
* Replay-equivalent — feeding the surviving log into the host
  reference (`qratum-arbiter`) reproduces the kernel's lock state.

## 2. On-disk layout

```
+--------------+--------------+--------------+
| Segment 0    | Segment 1    | Segment N    |
| (4 KiB)      | (4 KiB)      | (4 KiB)      |
+--------------+--------------+--------------+
```

Each segment:

```
+--------+-----------------------------------+----------+
| Header | Frames (variable, packed)         | Padding  |
| 32 B   |                                   | (zeros)  |
+--------+-----------------------------------+----------+
```

### Header (32 bytes)

| Off | Size | Field          | Notes                                       |
|----:|-----:|----------------|---------------------------------------------|
|   0 |    4 | magic          | `b"QAUD"` (0x44_55_41_51 LE)                |
|   4 |    2 | version        | `2`                                         |
|   6 |    2 | flags          | bit0=sealed, bit1=fenced                    |
|   8 |    8 | seq            | monotonic segment seq                       |
|  16 |    4 | frame_count    | frames in this segment                      |
|  20 |    4 | bytes_used     | bytes used in this segment incl. header     |
|  24 |    8 | prev_tail_h    | low 64 bits of Blake3 chain hash at start   |

### Frame

```
+-------+-------+--------+----------+
| len   | kind  | seq    | payload  |
| u16   | u8    | u32    | bytes    |
+-------+-------+--------+----------+
```

After each frame, the chaining state advances:

```
tail_hash = Blake3(tail_hash || frame_bytes)
```

The full 32-byte `tail_hash` is written into the segment **trailer**
once the segment is sealed:

### Trailer (32 bytes, written on seal)

| Off              | Size | Field              |
|-----------------:|-----:|--------------------|
| segment_end - 32 |   32 | tail_hash (Blake3) |

Until the trailer is written, the segment is `unsealed` and recovery
treats it as in-flight.

## 3. Write path (kernel)

1. `audit::append(kind, payload)` writes to in-memory MPSC ring (P0.2).
2. Background flusher locks no kernel paths; it copies a contiguous
   range of frames from the ring into the active segment buffer.
3. When `bytes_used + len > 4032` (4 KiB minus 32 B trailer) the
   flusher seals: writes the trailer, issues a barrier, then advances
   the active segment.
4. Sealed segments are the only durable state; unsealed bytes are
   advisory.

## 4. Recovery path (cold boot or replay)

1. Walk segments in `seq` order until first segment with bad magic or
   non-contiguous `seq`.
2. For each sealed segment: rederive the chain hash from its frames,
   compare to trailer. If divergent → discard segment and stop.
3. For trailing unsealed segment: walk frames, deriving the chain
   hash; the longest prefix whose internal cross-references remain
   consistent is recovered. Frames past the first inconsistency are
   torn writes.
4. Hand the recovered chain hash to `intent::recover(initial_state)`
   which replays into the host reference and asserts the resulting
   `LockState` matches the kernel's expected starting state.

## 5. Format compatibility

* v1 (P0.2) — in-memory only, no segments. Loaded as a single
  unsealed segment by recovery.
* v2 (P0.3) — this document.

## 6. Verification

* `crates/qratum-arbiter/tests/audit_recovery.rs` exercises items 2–4
  above against the host reference.
* The kernel-side flusher lives in
  `os/qratum-os/src/kernel/audit_persist.rs` (skeleton in P0.3,
  block-device hookup in P0.3.1).
