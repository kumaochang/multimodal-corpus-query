"""Draw the Study 3 sample from the recorded day frame.

Implements PREREGISTRATION.md section 1 item 2 and section 7 Amendments 1 and 2:
  - the day frame: dated entries whose 【】 bracket opens with the programme name and names
    the playlist's language, one entry per video id (Amendment 2), less Study 1's days and any
    additional exclusion that is present (Amendment 1);
  - the day draw: month-stratified when the frame spans at least the threshold number of
    months, otherwise uniform; all days taken if the frame is short (Amendment 1);
  - the stream: one default_rng(seed); every day draw first, languages alphabetically; then
    every within-day clip draw, languages alphabetically and days in date order (Amendment 1),
    with the calls fixed in Amendment 2.

Inputs, all from disk: --design, --frame. Output: --out (analysis/study3/draw.json).
The output carries no timestamp, so two runs on the same inputs are byte-identical.
"""
import argparse
import collections
import hashlib
import json
import re
import sys

import numpy as np

STUDY1_KEY = re.compile(r"^news-(?P<language>[A-Za-z]+)-(?P<date>\d{8})-\d{2}$")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def study1_days(path):
    days = set()
    for key in load_json(path):
        m = STUDY1_KEY.match(key)
        if not m:
            sys.exit(f"{path}: key {key!r} is not news-<Language>-<YYYYMMDD>-<NN>; stopping")
        days.add((m["language"], m["date"]))
    return days


def eligible_frame(name, lang, programme, excluded_days):
    """Apply Amendment 2, then the Amendment 1 exclusions. Returns (days, record)."""
    other_programme, other_language, by_date = [], [], collections.defaultdict(list)
    for r in lang["dated"]:
        if r["repeat_of_earlier_position"]:
            continue
        brief = {k: r[k] for k in ("position", "video_id", "broadcast_date", "title")}
        if not (r["bracket"] or "").startswith(programme):
            other_programme.append(brief)
        elif not r["bracket_names_language"]:
            other_language.append(brief)
        else:
            by_date[r["broadcast_date"]].append(r)

    multi = {d: [r["video_id"] for r in rs] for d, rs in by_date.items() if len(rs) > 1}
    if multi:
        sys.exit(f"{name}: dates carried by more than one video id after Amendment 2: {multi}; "
                 "stopping (Amendment 2)")

    excluded = []
    for d in sorted(by_date):
        if (name, d) in excluded_days:
            r = by_date.pop(d)[0]
            excluded.append({"video_id": r["video_id"], "broadcast_date": d, "title": r["title"],
                             "reason": excluded_days[(name, d)]})

    days = [{"broadcast_date": d, "video_id": by_date[d][0]["video_id"],
             "title": by_date[d][0]["title"], "duration_s": by_date[d][0]["duration_s"]}
            for d in sorted(by_date)]
    record = {
        "excluded_other_programme": len(other_programme),
        "excluded_other_language": other_language,
        "excluded_by_identity": excluded,
    }
    return days, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    design = {k: v["value"] for k, v in load_json(args.design).items()}
    frame = load_json(args.frame)

    excluded_days = {d: "Study 1 broadcast day" for d in study1_days(design["study1_days"])}
    extra_fired = []
    for x in design["additional_exclusions_if_present"]:
        key = (x["language"], x["broadcast_date"])
        if key not in excluded_days:
            excluded_days[key] = "additional exclusion, Amendment 1"

    n_days = design["days_per_language"]
    min_months = design["min_months_for_month_stratified_draw"]
    n_clips = design["clips_per_day"]
    k_clips = design["clips_drawn_per_day"]

    rng = np.random.default_rng(design["seed"])
    stream = []

    def consume(language, purpose, n, size=None):
        if size is None:
            out = [int(rng.choice(n))]
            call = f"rng.choice({n})"
        else:
            out = [int(i) for i in rng.choice(n, size=size, replace=False)]
            call = f"rng.choice({n}, size={size}, replace=False)"
        stream.append({"step": len(stream) + 1, "language": language, "purpose": purpose,
                       "call": call, "result": out})
        return out

    languages = sorted(frame["languages"])
    out_langs = {}

    # Stage 1: every day draw, languages alphabetically.
    for name in languages:
        days, record = eligible_frame(name, frame["languages"][name], design["programme"],
                                      excluded_days)
        for e in record["excluded_by_identity"]:
            if e["reason"] == "additional exclusion, Amendment 1":
                extra_fired.append({"language": name, "broadcast_date": e["broadcast_date"]})
        months = sorted({d["broadcast_date"][:6] for d in days})
        if len(days) < n_days:
            branch, drawn = "all days (frame shorter than required)", list(range(len(days)))
        elif len(months) >= min_months:
            branch = "month-stratified"
            picked = sorted(months[i] for i in consume(name, "months", len(months), n_days))
            drawn = []
            for mth in picked:
                in_month = [i for i, d in enumerate(days) if d["broadcast_date"][:6] == mth]
                drawn.append(in_month[consume(name, f"day in {mth}", len(in_month))[0]])
        else:
            branch = "uniform"
            drawn = consume(name, "days", len(days), n_days)
        drawn = sorted(drawn)
        out_langs[name] = {
            "frame": {"days": len(days), "months": len(months), "first": days[0]["broadcast_date"] if days else None,
                      "last": days[-1]["broadcast_date"] if days else None, **record,
                      "eligible_days": days},
            "branch": branch,
            "shortfall": max(0, n_days - len(days)),
            "drawn_days": [dict(days[i]) for i in drawn],
        }

    # Stage 2: every within-day clip draw, languages alphabetically, days in date order.
    for name in languages:
        for day in out_langs[name]["drawn_days"]:
            take = consume(name, f"clips of {day['broadcast_date']}", n_clips, min(k_clips, n_clips))
            day["clip_indices_drawn"] = sorted(i + 1 for i in take)

    result = {
        "rule": "PREREGISTRATION.md section 1 item 2; section 7 Amendments 1 and 2",
        "inputs": {
            "design": {"path": args.design.replace("\\", "/"), "sha256": sha256(args.design)},
            "frame": {"path": args.frame.replace("\\", "/"), "sha256": sha256(args.frame)},
            "study1_days": {"path": design["study1_days"], "sha256": sha256(design["study1_days"])},
        },
        "numpy_version": np.__version__,
        "design": design,
        "additional_exclusions_fired": extra_fired,
        "languages": out_langs,
        "stream": stream,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
