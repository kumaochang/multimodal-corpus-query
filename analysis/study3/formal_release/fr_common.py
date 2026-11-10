"""ULFR-v1 shared paths, hashing and content-addressed ranking storage. No analysis at import."""
import hashlib, json, os, sys, time, zipfile
from pathlib import Path

RELEASE_ID = 'ULFR-v1-20261025'
ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'analysis/study3/formal_release'
DATA = ROOT / 'data/study3_stats/formal_release/v1'
INPUT = ROOT / 'data/study3_runs'
PIPE = ROOT / 'pipeline/analysis'
VERIFIED = ROOT
CORE = ROOT
REF = CORE / 'analysis/study3/ch5_exploratory'
REFDATA = CORE / 'data/study3_stats/ch5_exploratory'
REPRO_REPORT = ROOT / 'analysis/study3/ch5_reproducibility'
REPRO_DATA = ROOT / 'data/study3_stats/ch5_reproducibility'
ITEM5 = Path(os.environ.get('RANKING_PYTHON', sys.executable))
POSTERIOR_PY = Path(os.environ.get('POSTERIOR_PYTHON', sys.executable))
CMDSTAN = REPRO_DATA / 'toolchain/cmdstan-2.37.0'
TREES = ROOT / 'work/data/aligned_trees'
POSE = ROOT / 'work/data/pose'
FROZEN = {
    'features_full/features.npz': '80ab8584dd41383226e0986055480ead9cac774b9e3b5928d4f2bf45d80d1704',
    'features_full/tfidf_sub.npz': '37166887f01dbe39e61b3ffde9ed0f746ba76f9e18d5f55c0eb2506ef6c71047',
    'features_full/tfidf_tr.npz': '2583b3af57b9bbaad01a5bf2056f3355335daa3b14affcf276a74bcffa43f029',
    'label_analysis/units.tsv': '0c5f9fd8c80bfbde1f6034b88d37145cd1b76c7684851bae0af6b0e92fdcf554',
    'unit_table/units_k.npz': 'd443a7566edd113a9ab6356797761de3a48bd427fffbc11d2f6c63451df38222',
    'ch5/ch5_retrieval.json': 'bd2d785d8b05fc96565df47ccb867795560b19e87650c7c5659379af5db9d4e2',
    'ch5/ch5_sensitivity.json': 'b39dba230d0d14f6c6e3a9d4d7edd0698c0563bb9dbfb09e22676f2eefe6c72f',
}
HISTORICAL = {  # L1 aggregates used for comparison only
    's62/s62_results.json': '3721fb6153f75f7d53572b8b550ee322b478e302c6213e6f1969c369d53e4b3c',
    's62/s62_by_label.json': 'fa15ab4fa2dae0ac17c5cca4cd7502c2cb47ce4bc1acac59ab6f4eb45ae275a2',
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, float) and x in (float('inf'), float('-inf')):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def dump(p, x, exclusive=True):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(clean(x), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def arrhash(x):
    import numpy as np
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(json.dumps([x.dtype.str, list(x.shape)]).encode())
    h.update(memoryview(x).cast('B') if x.size else b'')
    return h.hexdigest()


def matrix_hash(X):
    """Hash of a dense or scipy-sparse normalized matrix, layout-explicit."""
    import numpy as np
    if hasattr(X, 'tocsr'):
        X = X.tocsr()
        h = hashlib.sha256(json.dumps(['csr', X.dtype.str, list(X.shape)]).encode())
        for a in (X.indptr, X.indices, X.data):
            h.update(arrhash(a).encode())
        return h.hexdigest()
    return arrhash(np.asarray(X))


def check_frozen():
    got = {k: sha(INPUT / k) for k in FROZEN}
    assert got == FROZEN, 'frozen input hash mismatch'
    return got


def threads_one():
    from threadpoolctl import threadpool_info
    info = threadpool_info()
    for lib in info:
        assert lib['num_threads'] == 1, lib
    return info


def save_npz(path, arrays, level=1):
    """np.load-compatible NPZ with fixed timestamps (byte-deterministic) and fast DEFLATE."""
    import numpy as np
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.partial{os.getpid()}')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for k, v in arrays.items():
            info = zipfile.ZipInfo(k + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            for attr in ('_compresslevel', 'compress_level'):  # ZipInfo uses __slots__; name varies by Python
                try:
                    setattr(info, attr, level)
                except AttributeError:
                    pass
            info.external_attr = 0o644 << 16
            with z.open(info, 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(v), allow_pickle=False)
    os.replace(tmp, path)


class BlockStore:
    """Content-addressed ranking blocks. One block = one query set against one candidate bank.

    mode='write' saves payloads; mode='hash' only computes the same hashes (independent repeat).
    Block hash covers query rows, bank rows, exclusion rule, offsets, ranked rows and scores.
    """

    def __init__(self, root, mode):
        assert mode in ('write', 'hash')
        self.root, self.mode, self.index = Path(root), mode, {}

    @staticmethod
    def digest(query_rows, bank_rows, rule, offsets, rank_rows, scores):
        h = hashlib.sha256(json.dumps({'release': 'ULFR-v1 block v1', 'rule': rule}).encode())
        for a in (query_rows, bank_rows, offsets, rank_rows, scores):
            h.update(arrhash(a).encode())
        return h.hexdigest()

    def put(self, query_rows, bank_rows, rule, offsets, rank_rows, scores):
        import numpy as np
        query_rows = np.ascontiguousarray(query_rows, np.int32)
        bank_rows = np.ascontiguousarray(bank_rows, np.int32)
        offsets = np.ascontiguousarray(offsets, np.int64)
        rank_rows = np.ascontiguousarray(rank_rows, np.int32)
        scores = np.ascontiguousarray(scores)
        assert offsets[0] == 0 and offsets[-1] == len(rank_rows) == len(scores) and len(offsets) == len(query_rows) + 1
        h = self.digest(query_rows, bank_rows, rule, offsets, rank_rows, scores)
        meta = dict(queries=int(len(query_rows)), entries=int(len(rank_rows)), score_dtype=scores.dtype.str,
                    rule=rule, bank_size=int(len(bank_rows)))
        if h in self.index:
            self.index[h]['references'] += 1
            return h
        if self.mode == 'write':
            folder = self.root / 'blocks' / h[:2]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (h + '.npz')
            if not path.exists():
                save_npz(path, dict(query_rows=query_rows, bank_rows=bank_rows, offsets=offsets,
                                    rank_rows=rank_rows, rank_scores=scores))
            k = 10
            n = np.diff(offsets)
            top_rows = np.full((len(query_rows), k), -1, np.int32)
            top_scores = np.full((len(query_rows), k), np.nan, scores.dtype)
            for j in range(len(query_rows)):
                m = min(k, int(n[j]))
                top_rows[j, :m] = rank_rows[offsets[j]:offsets[j] + m]
                top_scores[j, :m] = scores[offsets[j]:offsets[j] + m]
            tfolder = self.root / 'topk10' / h[:2]
            tfolder.mkdir(parents=True, exist_ok=True)
            tpath = tfolder / (h + '.npz')
            if not tpath.exists():
                save_npz(tpath, dict(query_rows=query_rows, n_cand=n, top_rows=top_rows, top_scores=top_scores))
            meta.update(file=str(path.relative_to(self.root)), sha256=sha(path), bytes=path.stat().st_size,
                        topk_file=str(tpath.relative_to(self.root)), topk_sha256=sha(tpath))
        meta['references'] = 1
        self.index[h] = meta
        return h


def environment_record():
    import platform
    rec = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               env={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
               recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        import numpy, scipy
        rec.update(numpy=numpy.__version__, scipy=scipy.__version__)
        from threadpoolctl import threadpool_info
        rec['threadpools'] = threadpool_info()
        for lib in rec['threadpools']:
            if lib.get('filepath') and Path(lib['filepath']).exists():
                lib['binary_sha256'] = sha(lib['filepath'])
    except ImportError:
        pass
    try:
        rec['cpu_model'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
        flags = next(l.split(':', 1)[1].split() for l in open('/proc/cpuinfo') if l.startswith('flags'))
        rec['cpu_flags_subset'] = sorted(f for f in flags if f.startswith(('avx', 'fma', 'sse4')))
    except (OSError, StopIteration):
        pass
    return rec
