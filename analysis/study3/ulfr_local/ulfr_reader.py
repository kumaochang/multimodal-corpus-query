"""Local read-only access to the formal analysis release ULFR-v1-20261025.

Shared by the case figures and the query interface. It reads saved rankings, scores and
per-query metrics; it never computes a similarity, a normalization, a similarity matrix, a
ranking (no argsort), a metric or an interval, and it never falls back to the interface
ranking (iface-v1), the Linux reproduction snapshot or any earlier cache.

Verification that is actually performed (not only documented):
* the numeric package manifest and the writer package manifest name the same release;
* every file read from the numeric package is checked for size and SHA-256 against the
  numeric package manifest (index, unit rows, top-10 views, per-query metric files);
* the condition index names the release and pins the unit-row file, whose SHA-256 is checked;
* a top-10 view or full block is checked against the SHA-256 recorded for it in the
  condition index (topk_sha256 / sha256);
* unit rows agree, by unit ID, with the frozen units.tsv (whose hash the release manifest pins).

Full ranking blocks held outside the numeric package (block overlays):
* an overlay is declared in LOCAL_PATH_MAP.json (payload root, overlay manifest, package
  manifest and verification report, each with a pinned SHA-256); the original numeric package
  and its MANIFEST.json are not modified, and no block is copied into it;
* the overlay manifest, package manifest and verification report must name this release,
  the SHA-256 of the condition index actually loaded and the same canonical block list; the
  overlay file list must equal the verification report's file list;
* for every condition an overlay lists, the block identity, relative path, entry count and
  SHA-256 must equal the condition index entry;
* when a block is opened, its size and SHA-256 are checked against the overlay manifest (and
  so the index), its structure against the index (queries, entries, offsets, dtype), and its
  query order, candidate counts and first ten ranks/scores (bitwise) against the saved top-10
  view of the same condition;
* a condition with no block in the numeric package and none in any declared overlay raises
  BlockMissingError; a hash, release or identity failure raises ReleaseError. Nothing is
  substituted.

Limits enforced:
* the top-10 view answers ranks 1..10 only; any request beyond the saved view raises
  ViewLimitError instead of truncating, and never reads another source;
* a request for more ranks than the saved ranking holds raises ViewLimitError (callers that
  want "up to k" must ask for min(k, candidates) explicitly);
* an unknown condition or a query that the condition does not contain raises ConditionError.

Opened blocks may be kept in a bounded least-recently-used cache (block_cache_bytes); nothing
is decompressed at start-up.

The server's original manifests and archived programs are not modified. The mapping from
server paths to this repository is recorded in LOCAL_PATH_MAP.json beside this file.
"""
import csv
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RELEASE_ID = "ULFR-v1-20261025"
TOPK = 10
OVERLAY_SCHEMA = "ulfr-interface-block-overlay/1"


class ReleaseError(Exception):
    """Release identity, manifest or hash failure."""


class ConditionError(ReleaseError):
    """Unknown condition, or the query is not part of the condition."""


class ViewLimitError(ReleaseError):
    """Requested ranks beyond the saved top-10 view, or beyond the saved ranking."""


class BlockMissingError(ReleaseError):
    """The full ranking block is not available locally."""


class OverlayError(ReleaseError):
    """A declared block overlay fails its manifest, report or identity checks."""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_path_map(path=HERE / "LOCAL_PATH_MAP.json"):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("release") != RELEASE_ID:
        raise ReleaseError(f"path map is for {m.get('release')}, not {RELEASE_ID}")
    return m


class BlockCache:
    """Bounded LRU cache of opened FullBlock objects, by decompressed array bytes."""

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self._d = OrderedDict()
        self.bytes = 0
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "peak_bytes": 0, "max_bytes": self.max_bytes}

    def get(self, cid):
        b = self._d.get(cid)
        if b is None:
            self.stats["misses"] += 1
            return None
        self._d.move_to_end(cid)
        self.stats["hits"] += 1
        return b

    def put(self, cid, block):
        if cid in self._d:
            return
        self._d[cid] = block
        self.bytes += block.nbytes
        while self.bytes > self.max_bytes and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self.bytes -= old.nbytes
            self.stats["evictions"] += 1
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.bytes)

    def discard(self, cid):
        b = self._d.pop(cid, None)
        if b is not None:
            self.bytes -= b.nbytes

    def info(self):
        return dict(self.stats, blocks=len(self._d), bytes=self.bytes, conditions=list(self._d))


class Release:
    def __init__(self, path_map=None, require_overlays=(), block_cache_bytes=0):
        self.map = load_path_map() if path_map is None else path_map
        if self.map.get("release") != RELEASE_ID:
            raise ReleaseError(f"path map is for {self.map.get('release')}, not {RELEASE_ID}")
        roots = self.map["local_roots"]
        self.numeric = ROOT / roots["numeric_core"]
        self.writer = ROOT / roots["writer_metadata"]
        self.restore = ROOT / roots["frozen_restore_runs"]
        self.read_log = {}                     # repo-relative path -> {bytes, sha256, role}
        self._digests = {}                     # (abs path, size, mtime_ns) -> sha256, this process only
        self._lock = threading.RLock()
        nm = json.loads((self.numeric / "MANIFEST.json").read_text(encoding="utf-8"))
        wm_path = self.writer / "analysis/study3/formal_release/RELEASE_MANIFEST.json"
        self.release_manifest = json.loads(wm_path.read_text(encoding="utf-8"))
        if nm.get("release") != RELEASE_ID or self.release_manifest.get("release") != RELEASE_ID:
            raise ReleaseError(f"release ids differ: numeric {nm.get('release')}, writer {self.release_manifest.get('release')}")
        self._num_files = {f["path"]: f for f in nm["files"]}
        self._log(self.numeric / "MANIFEST.json", "numeric package manifest")
        self._log(wm_path, "writer release manifest")
        wpkg = json.loads((ROOT / "analysis/study3/ulfr_local/METADATA_MANIFEST.json").read_text(encoding="utf-8"))
        rel_wm = "analysis/study3/formal_release/RELEASE_MANIFEST.json"
        entry = next((f for f in wpkg["files"] if f["path"] == rel_wm), None)
        if entry is None or entry["sha256"] != self._digest(wm_path) or entry["bytes"] != wm_path.stat().st_size:
            raise ReleaseError("RELEASE_MANIFEST.json does not match the writer package manifest")
        ip = self._numeric_file("index/condition_index.json", "condition index")
        self.index_sha256 = self.read_log[self._rel(ip)]["sha256"]
        self.index = json.loads(ip.read_text(encoding="utf-8"))
        if self.index.get("release") != RELEASE_ID:
            raise ReleaseError(f"condition index release is {self.index.get('release')}")
        ur = self._numeric_file("index/unit_rows.npz", "unit rows")
        if self.read_log[self._rel(ur)]["sha256"] != self.index["unit_rows_sha256"]:
            raise ReleaseError("unit_rows.npz SHA-256 differs from the condition index")
        z = np.load(ur, allow_pickle=False)
        self.units = {k: z[k] for k in z.files}
        self.ids = self.units["unit_id"]
        self.row_of = {str(u): i for i, u in enumerate(self.ids)}
        if len(self.row_of) != len(self.ids):
            raise ReleaseError("duplicate unit IDs in unit_rows.npz")
        self._views = {}
        self._metrics = {}
        self._overlays = {}                    # name -> validated overlay record
        self._overlay_of = {}                  # condition -> (overlay name, file entry)
        self.cache = BlockCache(block_cache_bytes) if block_cache_bytes else None
        for name in require_overlays:
            self.overlay(name)

    # ---------------------------------------------------------------- files
    def _rel(self, p):
        p = Path(p).resolve()
        try:
            return str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:                    # outside the repository (test fixtures only)
            return str(p)

    def _digest(self, p):
        p = Path(p).resolve()
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
        d = self._digests.get(key)
        if d is None:
            d = sha256(p)
            self._digests[key] = d
        return d

    def _log(self, p, role, digest=None):
        p = Path(p)
        self.read_log[self._rel(p)] = {"bytes": p.stat().st_size, "sha256": digest or self._digest(p), "role": role}

    def _numeric_file(self, rel, role):
        """A file of the numeric package, verified against the package manifest."""
        e = self._num_files.get(rel)
        if e is None:
            raise ReleaseError(f"{rel} is not listed in the numeric package manifest")
        p = self.numeric / rel
        if not p.exists():
            raise ReleaseError(f"{rel} is listed but missing locally")
        d = self._digest(p)
        if p.stat().st_size != e["bytes"] or d != e["sha256"]:
            raise ReleaseError(f"{rel}: size/SHA-256 differ from the numeric package manifest")
        self._log(p, role, d)
        return p

    # ---------------------------------------------------------------- units
    def verify_unit_rows_against_units_tsv(self):
        """Unit rows equal the frozen units.tsv (hash pinned by the release) for every unit ID."""
        p = self.restore / "label_analysis/units.tsv"
        pinned = self.release_manifest["frozen_inputs"]["label_analysis/units.tsv"]
        d = self._digest(p)
        if d != pinned:
            raise ReleaseError("local units.tsv SHA-256 differs from the release frozen input")
        self._log(p, "frozen input units.tsv (release frozen_inputs)", d)
        cols = {"lang": "lang", "day": "day", "clip": "clip", "haiku": "haiku", "kept": "kept", "gap": "gap",
                "lid_answer": "lid_answer", "mf_class": "mf_class"}
        rows = {r["unit_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")}
        if set(rows) != set(self.row_of):
            raise ReleaseError("unit IDs in unit_rows.npz and units.tsv differ")
        bad = 0
        for i, u in enumerate(self.ids):
            r = rows[str(u)]
            for k, c in cols.items():
                if str(self.units[k][i]) != r[c]:
                    bad += 1
        if bad:
            raise ReleaseError(f"{bad} unit-row fields differ from units.tsv")
        return {"units": len(rows), "fields_compared": len(cols), "mismatches": 0}

    def unit(self, unit_id):
        i = self.row_of.get(unit_id)
        if i is None:
            raise ConditionError(f"unit {unit_id} is not in unit_rows.npz")
        return {"row": i, **{k: str(v[i]) for k, v in self.units.items()}}

    # ---------------------------------------------------------------- conditions
    def condition(self, cid):
        e = self.index["conditions"].get(cid)
        if e is None:
            raise ConditionError(f"condition '{cid}' is not in the {RELEASE_ID} condition index")
        return dict(e, condition_id=cid)

    def conditions(self, prefix=""):
        return [c for c in self.index["conditions"] if c.startswith(prefix)]

    def topk(self, cid):
        with self._lock:
            if cid in self._views:
                return self._views[cid]
            e = self.condition(cid)
            rel = "rankings/" + e["topk_file"]
            p = self._numeric_file(rel, "top-10 view")
            if self.read_log[self._rel(p)]["sha256"] != e["topk_sha256"]:
                raise ReleaseError(f"{cid}: top-10 file SHA-256 differs from the condition index")
            v = TopKView(self, cid, e, p)
            self._views[cid] = v
            return v

    # ---------------------------------------------------------------- overlays
    def overlay_names(self):
        return list(self.map.get("overlays", {}))

    def overlay(self, name):
        """Validate a declared block overlay (manifests and identities; payload hashes are
        checked when each block is opened, or all at once by verify_overlay_payloads)."""
        with self._lock:
            if name in self._overlays:
                return self._overlays[name]
            spec = self.map.get("overlays", {}).get(name)
            if spec is None:
                raise OverlayError(f"overlay '{name}' is not declared in the path map")

            def pinned(key_path, key_sha, role):
                p = ROOT / spec[key_path]
                if not p.exists():
                    raise OverlayError(f"overlay {name}: {spec[key_path]} is missing")
                d = self._digest(p)
                if d != spec[key_sha]:
                    raise OverlayError(f"overlay {name}: {spec[key_path]} SHA-256 {d[:12]} differs from the pinned {spec[key_sha][:12]}")
                self._log(p, role, d)
                return json.loads(p.read_text(encoding="utf-8"))

            man = pinned("manifest", "manifest_sha256", f"block overlay manifest ({name})")
            pkg = pinned("package_manifest", "package_manifest_sha256", f"block package manifest ({name})")
            rep = pinned("verification_report", "verification_report_sha256", f"block verification report ({name})")
            errs = []

            def need(ok, msg):
                if not ok:
                    errs.append(msg)
            need(man.get("schema") == OVERLAY_SCHEMA, f"overlay schema {man.get('schema')}")
            for what, doc in (("overlay manifest", man), ("package manifest", pkg), ("verification report", rep)):
                need(doc.get("release") == RELEASE_ID, f"{what} release {doc.get('release')}")
                need(doc.get("canonical_list_sha256") == spec["canonical_list_sha256"], f"{what} canonical list differs")
            need(man.get("index_sha256") == self.index_sha256 == spec["index_sha256"], "overlay manifest index SHA-256 differs from the loaded index")
            need(rep.get("index_sha256") == self.index_sha256, "verification report index SHA-256 differs from the loaded index")
            need(pkg.get("index_sha256_before") == pkg.get("index_sha256_after") == self.index_sha256, "package manifest index SHA-256 differs")
            need(man.get("payload_root") == spec["payload_root"], "overlay payload root differs from the path map")
            need(rep.get("isolation_root") == spec["payload_root"], "verification report root differs from the path map")
            need(man.get("package_manifest") == spec["package_manifest"], "overlay manifest names another package manifest")
            need(man.get("package_manifest_sha256") == spec["package_manifest_sha256"]
                 == rep.get("package_manifest_sha256"), "package manifest SHA-256 differs between documents")
            need(rep.get("status") == "PASS", f"verification report status {rep.get('status')}")
            files = man.get("files", [])
            keyf = lambda f: (f["path"], f["bytes"], f["sha256"], f["block"], f["entries"], tuple(f["conditions"]))
            need(sorted(map(keyf, files)) == sorted(map(keyf, rep.get("files", []))), "overlay file list differs from the verification report")
            need(len(files) == pkg.get("payload_count") == rep.get("payload_files"), "payload count differs")
            need(sum(f["bytes"] for f in files) == pkg.get("payload_bytes") == rep.get("payload_bytes"), "payload bytes differ")
            need(sum(f["entries"] for f in files) == pkg.get("entries") == rep.get("entries"), "entry total differs")
            need(pkg.get("missing") == 0 and pkg.get("duplicate_or_extra") == 0, "package manifest reports missing or extra payloads")
            if spec.get("canonical_list"):
                cp = ROOT / spec["canonical_list"]
                if not cp.exists():
                    raise OverlayError(f"overlay {name}: canonical list {spec['canonical_list']} is missing")
                cd = self._digest(cp)
                need(cd == spec["canonical_list_sha256"], "canonical list file SHA-256 differs from the pinned value")
                self._log(cp, f"canonical block list ({name})", cd)
                canon = sorted(tuple(line.split("\t")) for line in cp.read_text(encoding="utf-8").splitlines() if line)
                need(canon == sorted((f["path"], f["sha256"], str(f["entries"]), ";".join(f["conditions"])) for f in files),
                     "overlay file list differs from the canonical block list")
            by_cond = {}
            for f in files:
                need(f["path"] == f"rankings/blocks/{f['block'][:2]}/{f['block']}.npz", f"{f['path']}: path does not match block identity")
                for c in f["conditions"]:
                    e = self.index["conditions"].get(c)
                    if e is None:
                        need(False, f"{f['path']}: condition {c} is not in the index")
                        continue
                    need(e["block"] == f["block"] and "rankings/" + e["file"] == f["path"] and e["sha256"] == f["sha256"]
                         and e["entries"] == f["entries"], f"{c}: overlay entry differs from the condition index")
                    need(c not in by_cond, f"{c}: listed twice")
                    by_cond[c] = f
            if errs:
                raise OverlayError(f"overlay {name}: " + "; ".join(errs[:8]) + (f" (+{len(errs) - 8} more)" if len(errs) > 8 else ""))
            for c in by_cond:
                if c in self._overlay_of and self._overlay_of[c][0] != name:
                    raise OverlayError(f"{c}: provided by overlays {self._overlay_of[c][0]} and {name}")
            rec = {"name": name, "root": ROOT / spec["payload_root"], "spec": spec, "files": files, "conditions": len(by_cond),
                   "payload_bytes": pkg["payload_bytes"], "entries": pkg["entries"]}
            for c, f in by_cond.items():
                self._overlay_of[c] = (name, f)
            self._overlays[name] = rec
            return rec

    def verify_overlay_payloads(self, name, progress=None):
        """Size and SHA-256 of every payload in an overlay against its manifest (reads ~all bytes)."""
        ov = self.overlay(name)
        bad = []
        for i, f in enumerate(ov["files"]):
            p = ov["root"] / f["path"]
            if not p.exists() or p.stat().st_size != f["bytes"] or self._digest(p) != f["sha256"]:
                bad.append(f["path"])
            if progress:
                progress(i + 1, len(ov["files"]))
        return {"files": len(ov["files"]), "bad": bad}

    def block_location(self, cid):
        """Where the full block of a condition is read from, without opening it."""
        e = self.condition(cid)
        p = self.numeric / "rankings" / e["file"]
        if p.exists():
            return {"source": "numeric_core", "path": self._rel(p), "sha256": e["sha256"]}
        for name in self.overlay_names():
            self.overlay(name)
        hit = self._overlay_of.get(cid)
        if hit is None:
            raise BlockMissingError(f"{cid}: full ranking block {e['file']} is not available locally "
                                    f"({e['entries']:,} entries; not in the numeric package or any declared overlay)")
        name, f = hit
        return {"source": f"overlay:{name}", "path": self._rel(self._overlays[name]["root"] / f["path"]),
                "sha256": f["sha256"], "bytes": f["bytes"]}

    def block(self, cid):
        with self._lock:
            if self.cache is not None:
                b = self.cache.get(cid)
                if b is not None:
                    return b
            e = self.condition(cid)
            loc = self.block_location(cid)
            p = ROOT / loc["path"]
            if not p.exists():
                raise BlockMissingError(f"{cid}: block file {loc['path']} is missing")
            if "bytes" in loc and p.stat().st_size != loc["bytes"]:
                raise ReleaseError(f"{cid}: block size differs from the overlay manifest")
            d = self._digest(p)
            if d != e["sha256"] or d != loc["sha256"]:
                raise ReleaseError(f"{cid}: block SHA-256 differs from the condition index")
            self._log(p, "full ranking block" + ("" if loc["source"] == "numeric_core" else f" ({loc['source']})"), d)
            b = FullBlock(self, cid, e, p, source=loc["source"])
            b.check_against_view(self.topk(cid))
            if self.cache is not None:
                self.cache.put(cid, b)
            return b

    # ---------------------------------------------------------------- metrics
    def metrics(self, cid, unit_id):
        """Saved per-query metrics for one query (supplement metric file or registered per-query file)."""
        e = self.condition(cid)
        rel = e.get("metrics_file") or e.get("perquery_file")
        if not rel:
            raise ConditionError(f"{cid}: no per-query metrics file in the index")
        rel = rel if rel.startswith(("supplement/", "registered/")) else "rankings/" + rel
        p = self._numeric_file(rel, "per-query metrics")
        z = np.load(p, allow_pickle=False)
        row = self.row_of.get(unit_id)
        if "q" not in z.files:
            raise ConditionError(f"{cid}: metrics file has no query key 'q'")
        hits = np.flatnonzero(z["q"] == row)
        if len(hits) != 1:
            raise ConditionError(f"{cid}: query {unit_id} found {len(hits)} times in {rel}")
        j = int(hits[0])
        out = {"file": self._rel(p), "position": j}
        for k in z.files:
            v = z[k][j]
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        return out

    def metric_fields(self, cid, unit_id, fields=("n_cand", "n_rel")):
        """Selected saved per-query metric fields, with a small per-file cache of those fields only."""
        e = self.condition(cid)
        rel = e.get("metrics_file") or e.get("perquery_file")
        if not rel:
            raise ConditionError(f"{cid}: no per-query metrics file in the index")
        rel = rel if rel.startswith(("supplement/", "registered/")) else "rankings/" + rel
        with self._lock:
            key = (rel, tuple(fields))
            m = self._metrics.get(key)
            if m is None:
                p = self._numeric_file(rel, "per-query metrics")
                z = np.load(p, allow_pickle=False)
                if "q" not in z.files:
                    raise ConditionError(f"{cid}: metrics file has no query key 'q'")
                q = z["q"]
                pos = {}
                for j, r in enumerate(q.tolist()):
                    pos.setdefault(r, []).append(j)
                m = {"file": self._rel(p), "pos": pos, **{f: z[f] for f in fields}}
                self._metrics[key] = m
        row = self.row_of.get(unit_id)
        hits = m["pos"].get(row, [])
        if len(hits) != 1:
            raise ConditionError(f"{cid}: query {unit_id} found {len(hits)} times in {rel}")
        j = hits[0]
        return {"file": m["file"], "position": j, **{f: m[f][j].item() for f in fields}}


class TopKView:
    """The saved top-10 view of one condition. Ranks beyond 10 are refused."""

    def __init__(self, rel, cid, e, path):
        z = np.load(path, allow_pickle=False)
        self.rel, self.cid, self.entry, self.path = rel, cid, e, path
        self.query_rows, self.n_cand = z["query_rows"], z["n_cand"]
        self.top_rows, self.top_scores = z["top_rows"], z["top_scores"]
        if self.top_rows.shape[1] != TOPK:
            raise ReleaseError(f"{cid}: top view has {self.top_rows.shape[1]} columns, expected {TOPK}")
        self._pos = {int(r): j for j, r in enumerate(self.query_rows)}

    def position(self, unit_id):
        row = self.rel.row_of.get(unit_id)
        if row is None or row not in self._pos:
            raise ConditionError(f"{self.cid}: query {unit_id} is not a query of this condition")
        return self._pos[row]

    def has_query(self, unit_id):
        row = self.rel.row_of.get(unit_id)
        return row is not None and row in self._pos

    def candidates(self, unit_id):
        return int(self.n_cand[self.position(unit_id)])

    def neighbors(self, unit_id, k):
        if not isinstance(k, int) or k < 1:
            raise ViewLimitError(f"k must be a positive integer, got {k!r}")
        if k > TOPK:
            raise ViewLimitError(f"{self.cid}: ranks 1..{k} requested, but only the saved top-{TOPK} view is local; "
                                 "the full ranking block is required")
        j = self.position(unit_id)
        n = int(self.n_cand[j])
        if k > n:
            raise ViewLimitError(f"{self.cid}: ranks 1..{k} requested, but this query has only {n} candidates")
        out = []
        for r in range(k):
            row = int(self.top_rows[j, r])
            if row < 0 or not np.isfinite(self.top_scores[j, r]):
                raise ReleaseError(f"{self.cid}: rank {r + 1} is empty in the saved view although n_cand={n}")
            out.append({"rank": r + 1, "unit_id": str(self.rel.ids[row]), "row": row,
                        "score": float(self.top_scores[j, r]), "score_dtype": str(self.top_scores.dtype),
                        "view_position": [j, r]})
        return out


class FullBlock:
    """A complete saved ranking, from the numeric package or a verified overlay."""

    def __init__(self, rel, cid, e, path, source="numeric_core"):
        z = np.load(path, allow_pickle=False)
        self.rel, self.cid, self.entry, self.path, self.source = rel, cid, e, path, source
        self.query_rows, self.bank_rows = z["query_rows"], z["bank_rows"]
        self.offsets, self.rank_rows, self.rank_scores = z["offsets"], z["rank_rows"], z["rank_scores"]
        q, off = self.query_rows, self.offsets
        errs = []
        if len(q) != e["queries"]:
            errs.append(f"{len(q)} queries, index {e['queries']}")
        if len(off) != len(q) + 1 or int(off[0]) != 0 or int(off[-1]) != e["entries"] or np.any(np.diff(off) < 0):
            errs.append("offsets inconsistent with the index entry count")
        if len(self.rank_rows) != e["entries"] or len(self.rank_scores) != e["entries"]:
            errs.append("rank arrays differ from the index entry count")
        if np.dtype(self.rank_scores.dtype).str != e["score_dtype"]:
            errs.append(f"score dtype {self.rank_scores.dtype.str}, index {e['score_dtype']}")
        n = len(rel.ids)
        if len(q) and (q.min() < 0 or q.max() >= n) or len(self.rank_rows) and (self.rank_rows.min() < 0 or self.rank_rows.max() >= n):
            errs.append("row indices outside unit rows")
        if errs:
            raise ReleaseError(f"{cid}: block structure: " + "; ".join(errs))
        self._pos = {int(r): j for j, r in enumerate(q)}
        self.nbytes = int(sum(a.nbytes for a in (self.query_rows, self.bank_rows, self.offsets, self.rank_rows, self.rank_scores)))
        self.view_check = None

    def check_against_view(self, v):
        """Query order, candidate counts and ranks 1..10 (rows and bitwise scores) equal the saved view."""
        if not np.array_equal(self.query_rows, v.query_rows):
            raise ReleaseError(f"{self.cid}: block query order differs from the saved top-10 view")
        n = np.diff(self.offsets)
        if not np.array_equal(n, v.n_cand.astype(n.dtype)):
            raise ReleaseError(f"{self.cid}: block candidate counts differ from the saved top-10 view")
        compared = 0
        for r in range(TOPK):
            has = n > r
            idx = self.offsets[:-1][has] + r
            if not np.array_equal(self.rank_rows[idx], v.top_rows[has, r].astype(self.rank_rows.dtype)):
                raise ReleaseError(f"{self.cid}: rank {r + 1} rows differ between block and top-10 view")
            a = np.ascontiguousarray(self.rank_scores[idx]).view(np.uint32)
            b = np.ascontiguousarray(v.top_scores[has, r].astype(self.rank_scores.dtype)).view(np.uint32)
            if not np.array_equal(a, b):
                raise ReleaseError(f"{self.cid}: rank {r + 1} scores differ between block and top-10 view")
            if np.any(v.top_rows[~has, r] >= 0):
                raise ReleaseError(f"{self.cid}: top-10 view holds a rank beyond the candidate count")
            compared += int(has.sum())
        self.view_check = {"queries": int(len(n)), "ranks_compared": compared}
        return self.view_check

    def has_query(self, unit_id):
        row = self.rel.row_of.get(unit_id)
        return row is not None and row in self._pos

    def candidates(self, unit_id):
        row = self.rel.row_of.get(unit_id)
        if row not in self._pos:
            raise ConditionError(f"{self.cid}: query {unit_id} is not a query of this condition")
        j = self._pos[row]
        return int(self.offsets[j + 1] - self.offsets[j])

    def neighbors(self, unit_id, k=None):
        row = self.rel.row_of.get(unit_id)
        if row not in self._pos:
            raise ConditionError(f"{self.cid}: query {unit_id} is not a query of this condition")
        j = self._pos[row]
        a, b = int(self.offsets[j]), int(self.offsets[j + 1])
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ViewLimitError(f"k must be a positive integer, got {k!r}")
        if k is not None and k > b - a:
            raise ViewLimitError(f"{self.cid}: ranks 1..{k} requested, but the saved ranking has {b - a} candidates")
        b = b if k is None else a + k
        return [{"rank": i + 1, "unit_id": str(self.rel.ids[r]), "row": int(r), "score": float(s),
                 "score_dtype": str(self.rank_scores.dtype), "block_position": a + i}
                for i, (r, s) in enumerate(zip(self.rank_rows[a:b], self.rank_scores[a:b]))]
