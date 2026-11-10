# Paired numerical baseline

Fused retrieval uses the fixed local float32 full-384 multiplication: per-block L2 normalization, concatenation and final L2 normalization. Exact ties retain original candidate order. Single-block controls from the same numerical lineage are the paired baselines. Every saved one-block full score matrix, ranking in all four pools and per-query metric is reproduced before multi-block calculations. PCG64 seed 2026110501 generates 10,000 shared day draws, 420 pooled P@1/P@3/MAP pointwise intervals and the fixed twelve-coordinate W3 simultaneous family.

Prosodic augmentation uses the saved local text-only baseline with the same sources, folds, train-only transformations and fixed regularization. T+prosody and T+Whisper use seed 2026110502 and 10,000 shared whole-day OOF evaluation draws, without refitting inside bootstrap. Historical server estimates remain separate.
