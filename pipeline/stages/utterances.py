"""The utterance inventory every Study 3 stage shares.

Utterances are the non-empty intervals of the `utterances` tier of the MOCO aligned tree
(`<clip>.aligntree.TextGrid`), the "existing utterance boundaries" of PREREGISTRATION.md. Their
labels are Whisper text, unusable on this material (analysis/q17), and are never read here: only
the times are. Each utterance's id is `<clip>_u<index>`, the index counting non-empty intervals
from 0 in time order, so every stage (subtitle assignment, transcription, prosody) names the
same utterance the same way.
"""
import re
from pathlib import Path

_HEAD = re.compile(r'"IntervalTier"\n"([^"]*)"\n([-\d.]+)\n([-\d.]+)\n(\d+)\n')

def read_tier(textgrid_path, name="utterances"):
    """Intervals (start, end, label) of one tier of a long-format TextGrid."""
    t = Path(textgrid_path).read_text(encoding="utf-8")
    for m in _HEAD.finditer(t):
        if m.group(1) != name:
            continue
        n = int(m.group(4))
        lines = t[m.end():].split("\n")
        out = []
        for k in range(n):
            a, b, lab = lines[3 * k], lines[3 * k + 1], lines[3 * k + 2]
            out.append((float(a), float(b), lab[1:-1] if lab.startswith('"') else lab))
        return out
    raise KeyError(f"tier {name!r} not in {textgrid_path}")

def load_utterances(textgrid_path, clip_id):
    """[(utt_id, start_s, end_s)] for the non-empty intervals, in time order."""
    rows = [(a, b) for a, b, lab in read_tier(textgrid_path) if lab.strip()]
    rows.sort()
    return [(f"{clip_id}_u{i:04d}", a, b) for i, (a, b) in enumerate(rows)]
