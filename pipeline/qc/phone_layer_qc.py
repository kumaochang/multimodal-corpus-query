"""Phone-layer QC: one acoustic pass per file, decoded under several inventories.

Configurations
  cmn        Mandarin inventory: what moco2026/new_make_allosaurus.py hard-codes (lang_id="cmn")
  cmn_emit2  the same at emit=2, the setting that script uses, to check the existing data/emit2 layer
  ipa        unrestricted (the full uni2005 set)
  shipped    the inventory Allosaurus ships for the language, where one exists (pwn, trv)
  grammar    the inventory derived from the 語法概論 phonology chapter (inventories/<iso>.txt)

Usage: python phone_layer_qc.py <manifest.json> <out.jsonl> <inventories dir>
manifest: [{"id":..., "wav":..., "iso":..., "set":"study1"|"ntu"}]
"""
import json, sys
from pathlib import Path
import numpy as np
from allosaurus.app import read_recognizer
from allosaurus.audio import read_audio
from allosaurus.am.utils import move_to_tensor

HERE = Path(__file__).resolve().parent
SHIPPED = {"pwn": "pwn", "trv": "trv", "tay": "tay", "tsu": "tsu"}
rec = read_recognizer()

def lprobs(wav):
    feat = rec.pm.compute(read_audio(str(wav)))
    feats = np.expand_dims(feat, 0)
    flen = np.array([feat.shape[0]], dtype=np.int32)
    t, tl = move_to_tensor([feats, flen], rec.config.device_id)
    return rec.am(t, tl).detach().numpy()[0]

def decode(lp, lang, emit=1.0):
    return rec.lm.compute(lp.copy(), lang, 1, emit=emit).split()

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
done = set()
if out.exists():
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
with out.open("a", encoding="utf-8") as fh:
    for m in manifest:
        if m["id"] in done:
            continue
        lp = lprobs(m["wav"])
        r = {"id": m["id"], "iso": m["iso"], "set": m["set"], "frames": int(lp.shape[0])}
        r["cmn"] = decode(lp, "cmn")
        r["cmn_emit2"] = decode(lp, "cmn", 2.0)
        r["ipa"] = decode(lp, "ipa")
        if m["iso"] in SHIPPED:
            r["shipped"] = decode(lp, SHIPPED[m["iso"]])
        r["grammar"] = decode(lp, str(Path(sys.argv[3]) / f"{m['iso']}.txt"))
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        print(m["id"], r["frames"], len(r["grammar"]), flush=True)
