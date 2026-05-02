# Replay of External Feeds

External data is ingested via sandbox sessions that record tick-ordered payloads and normalized outputs. These histories can be replayed through `qsk.distributed.real_mirror.mirror_replay` to reconstruct mirror states and compare them against cluster world states using `mirror_diff`.

Replay ensures deterministic forensics and enables scenario what-if analysis grounded in real-world datasets.
