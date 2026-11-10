"""Build the Study 3 day frame from TITV playlist metadata.

PREREGISTRATION.md section 7, Amendment 1: the frame is built from playlist metadata alone,
with no video downloaded and no content viewed. Every entry whose title carries a parsable
eight-digit broadcast date is a dated entry; entries whose date does not parse are listed.

This script records what each playlist holds. It makes no eligibility decision beyond the
date parse: entries are flagged (repeated id, more than one video on a date, title language
not the playlist's language), never dropped.

Inputs, all from disk:
  --playlists  JSON naming each language, its Chinese name and its playlist URL
               (analysis/study3/playlists.json)
Output:
  --out        analysis/study3/frame.json

yt-dlp is invoked as the YTDLP environment variable says (split on whitespace), default
"yt-dlp". The listing must be complete: if the number of entries returned differs from the
playlist's own count, the script stops. yt-dlp 2026.03.03 returns only the first 100.
"""
import argparse
import collections
import datetime
import json
import os
import re
import shlex
import subprocess
import sys

DATE_RUN = re.compile(r"(?<!\d)\d{8}(?!\d)")
BRACKET = re.compile(r"【([^】]*)】")


def ytdlp_cmd():
    return shlex.split(os.environ.get("YTDLP", "yt-dlp"), posix=False)


def ytdlp_version():
    out = subprocess.run(ytdlp_cmd() + ["--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def list_playlist(url):
    cmd = ytdlp_cmd() + ["--no-update", "--flat-playlist", "-J",
                         "--extractor-args", "youtube:lang=zh-TW", url]
    out = subprocess.run(cmd, capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8")), cmd


def parse_date(title):
    runs = DATE_RUN.findall(title)
    if len(runs) != 1:
        return None, f"{len(runs)} eight-digit runs"
    s = runs[0]
    try:
        datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None, "not a calendar date"
    return s, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlists", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.playlists, encoding="utf-8") as f:
        config = json.load(f)

    frame = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yt_dlp_version": ytdlp_version(),
        "playlists_file": args.playlists.replace("\\", "/"),
        "rule": "PREREGISTRATION.md section 7 Amendment 1: one dated entry per title carrying "
                "exactly one eight-digit run that is a calendar date; all others listed as unparsed",
        "languages": {},
    }

    for lang in config["languages"]:
        name, zh, url = lang["language"], lang["zh"], lang["playlist"]
        data, cmd = list_playlist(url)
        entries = data.get("entries") or []
        if len(entries) != data.get("playlist_count"):
            sys.exit(f"{name}: {len(entries)} entries returned, playlist_count "
                     f"{data.get('playlist_count')}; listing incomplete, stopping")

        seen = collections.Counter()
        dated, unparsed = [], []
        for pos, e in enumerate(entries, start=1):
            title = e.get("title") or ""
            seen[e["id"]] += 1
            m = BRACKET.search(title)
            bracket = m.group(1) if m else None
            rec = {
                "position": pos,
                "video_id": e["id"],
                "title": title,
                "duration_s": e.get("duration"),
                "bracket": bracket,
                "programme": re.split(r"[-－ ]", bracket)[0] if bracket else None,
                "bracket_names_language": bool(bracket and zh in bracket),
                "repeat_of_earlier_position": seen[e["id"]] > 1,
            }
            date, why = parse_date(title)
            if date:
                rec["broadcast_date"] = date
                dated.append(rec)
            else:
                rec["unparsed_reason"] = why if title else "no title returned"
                unparsed.append(rec)

        ids_by_date = collections.defaultdict(set)
        for r in dated:
            ids_by_date[r["broadcast_date"]].add(r["video_id"])
        dates = sorted(ids_by_date)
        months = sorted({d[:6] for d in dates})
        frame["languages"][name] = {
            "zh": zh,
            "playlist_url": url,
            "playlist_id": data.get("id"),
            "playlist_title": data.get("title"),
            "playlist_count": data.get("playlist_count"),
            "yt_dlp_command": " ".join(cmd),
            "summary": {
                "entries": len(entries),
                "dated_entries": len(dated),
                "unparsed_entries": len(unparsed),
                "distinct_dates": len(dates),
                "distinct_months": len(months),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
                "dates_with_more_than_one_video_id": sorted(d for d, v in ids_by_date.items() if len(v) > 1),
                "dated_entries_not_naming_language": [r["position"] for r in dated if not r["bracket_names_language"]],
                "programme_counts": dict(collections.Counter(r["programme"] for r in dated
                                                             if not r["repeat_of_earlier_position"]).most_common()),
            },
            "dated": dated,
            "unparsed": unparsed,
        }
        print(f"{name}: {len(entries)} entries, {len(dated)} dated, {len(unparsed)} unparsed, "
              f"{len(dates)} dates over {len(months)} months", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(frame, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
