"""Writing tables from accepted-recipe saved results only, without reranking/refitting."""

import csv
import json
from itertools import combinations
from pathlib import Path

from execution_support import record, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROSODY = HERE.parent / "prosody_preparation"


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    f1 = json.loads((ROOT / "data/study3_stats/ch5_fusion/fusion_local/RESULTS.json").read_bytes())
    p1 = json.loads((ROOT / "data/study3_stats/prosody_exploration/P1_augmented_local/RESULTS.json").read_bytes())
    f0path = ROOT / "analysis/clip_level_map_audit/fusion15_results.json"
    f0pin = record(f0path)
    f0 = json.loads(f0path.read_bytes())
    assert json.loads((HERE / "FUSION_VALIDATION.json").read_bytes())["status"] == "PASS"
    assert json.loads((PROSODY / "P1_AUGMENTED_VALIDATION.json").read_bytes())["status"] == "PASS"
    long = []
    wide = {}
    for row in f1["summary_rows"]:
        r = dict(row)
        r["ci95_low"],r["ci95_high"] = r.pop("ci95_pointwise")
        long.append(r)
        key = (r["configuration"],r["pool"],r["group"])
        w = wide.setdefault(key,{k:r[k] for k in ("configuration","pool","version","subset","group","language_scope","queries","candidate_min","candidate_max","no_candidates","zero_relevant","coverage_status")})
        for k in ("estimate","ci95_low","ci95_high","effective_n","days","clips","undefined_draws"):
            w[r["metric"]+"_"+k]=r[k]
    for w in wide.values():
        w["P1_chance"]=w["chance_estimate"]
        w["P3_chance"]=w["chance_estimate"]
    save_csv(HERE/"FUSION_RESULTS.csv",long)
    save_csv(HERE/"FUSION_WIDE.csv",list(wide.values()))
    contrast=[]
    for row in f1["contrasts"]:
        r={k:v for k,v in row.items() if k not in ("ci95_pointwise","ci95_simultaneous")}
        r["pointwise_low"],r["pointwise_high"]=row["ci95_pointwise"]
        r["simultaneous_low"],r["simultaneous_high"]=row["ci95_simultaneous"] if row["ci95_simultaneous"] is not None else (None,None)
        contrast.append(r)
    save_csv(HERE/"FIXED_CONTRASTS.csv",contrast)
    bindings={}
    for text in ("xlmr_sub","tfidf_sub"):
        for audio in ("whisper","mfcc86"):
            blocks=[text,audio,"prosody","kin14"]
            configs=["+".join(c) for n in range(1,5) for c in combinations(blocks,n)]
            assert len(configs)==15 and all((c,"W3","all") in wide for c in configs)
            bindings[text+"/"+audio]=configs
    write_json(HERE/"METHOD_SET_BINDINGS.json",{"sets":bindings,"unique_configurations":35,"no_duplicate_computation":True})
    old={r["configuration"]:r for r in f0["results"]}
    symbols={"text":"T","audio":"A","prosody":"P","kinematic":"K"}
    common_cols=["stage","method_set","configuration","source_configuration","pool","query_group","queries","candidate_min","candidate_max","P1_n","P3_n","MAP_n","P1","P3","MAP","P1_minus_T","P3_minus_T","MAP_minus_T","interval_status"]
    stage=[]
    for name,r in old.items():
        stage.append(dict(zip(common_cols,["clip","subtitle_xlmr/MFCC86/phone8/K18","+".join(symbols[x] for x in name.split("+")),name,"all_other_clips","all",30,29,29,30,30,30,r["P1"],r["P3"],r["MAP"],r["P1"]-old["text"]["P1"],r["P3"]-old["text"]["P3"],r["MAP"]-old["text"]["MAP"],"descriptive point only"])))
    for method,configs in bindings.items():
        text,audio=method.split("/")
        mapping={text:"T",audio:"A","prosody":"P","kin14":"K"}
        for name in configs:
            for pool in ("W1","W2","W3","X"):
                r,b=wide[(name,pool,"all")],wide[(text,pool,"all")]
                stage.append(dict(zip(common_cols,["utterance",method+"/prosody15/K14","+".join(mapping[x] for x in name.split("+")),name,pool,"all",r["queries"],r["candidate_min"],r["candidate_max"],r["P1_effective_n"],r["P3_effective_n"],r["MAP_effective_n"],r["P1_estimate"],r["P3_estimate"],r["MAP_estimate"],r["P1_estimate"]-b["P1_estimate"],r["P3_estimate"]-b["P3_estimate"],r["MAP_estimate"]-b["MAP_estimate"],"pointwise estimates and CIs in FUSION_WIDE.csv; fixed-family intervals in FIXED_CONTRASTS.csv"])))
    save_csv(HERE/"STAGE_COMPARISON.csv",stage)
    lines=["# Primary method set, W3", "", "Subtitle XLM-R / Whisper / prosody15 / K14, all-query common384. Point estimates below use the adopted local one-block controls. Display rounding is declared here only; full precision and intervals remain in the CSV files.", "",
           "| Configuration | P@1 | P@3 | MAP |", "|---|---:|---:|---:|"]
    for name in bindings["xlmr_sub/whisper"]:
        r=wide[(name,"W3","all")]
        lines.append(f"| {name} | {r['P1_estimate']:.6f} | {r['P3_estimate']:.6f} | {r['MAP_estimate']:.6f} |")
    lines += ["", "# Fixed paired contrasts", "", "T is subtitle XLM-R. Intervals below are the simultaneous twelve-coordinate family, not pointwise intervals.", "",
              "| Contrast | Metric | Difference | Simultaneous 95% interval |", "|---|---|---:|---|"]
    for row in contrast:
        ci="undefined family" if row['simultaneous_low'] is None else f"[{row['simultaneous_low']:.6f}, {row['simultaneous_high']:.6f}]"
        lines.append(f"| {row['configuration']} minus T | {row['metric']} | {row['estimate']:.6f} | {ci} |")
    (HERE/"RESULTS_TABLES.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save_csv(PROSODY/"P1_MODELS.csv",[{**{k:v for k,v in r.items() if k!='ci95_pointwise'},"ci95_low":r['ci95_pointwise'][0],"ci95_high":r['ci95_pointwise'][1]} for r in p1['models']])
    prows=[]
    for r in p1['contrasts']:
        prows.append({**{k:v for k,v in r.items() if k not in ('ci95_pointwise','ci95_simultaneous')},
                      "pointwise_low":r['ci95_pointwise'][0],"pointwise_high":r['ci95_pointwise'][1],
                      "simultaneous_low":r['ci95_simultaneous'][0] if r['ci95_simultaneous'] else None,
                      "simultaneous_high":r['ci95_simultaneous'][1] if r['ci95_simultaneous'] else None})
    save_csv(PROSODY/"P1_CONTRASTS.csv",prows)
    lines=["# Prosody and Whisper channel separation", "", "Primary192 / positive20 / 12 days / 78 clips. All paired comparisons use the adopted local T; old registered model values are preserved separately. AP is weighted average precision, not retrieval MAP.", "",
           "| Model | AUC | AP |", "|---|---:|---:|"]
    for name in ("local_T","T_plus_P","T_plus_W"):
        x={r['metric']:r['estimate'] for r in p1['models'] if r['model']==name}
        lines.append(f"| {name} | {x['AUC']:.6f} | {x['AP']:.6f} |")
    lines += ["", "| Contrast | Metric | Difference | Pointwise 95% interval | Simultaneous 95% interval |", "|---|---|---:|---|---|"]
    for r in prows:
        lines.append(f"| {r['model']} minus {r['baseline']} | {r['metric']} | {r['estimate']:.6f} | [{r['pointwise_low']:.6f}, {r['pointwise_high']:.6f}] | [{r['simultaneous_low']:.6f}, {r['simultaneous_high']:.6f}] |")
    (PROSODY/"P1_RESULTS_TABLE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    assert record(f0path)==f0pin
    write_json(HERE/"TABLE_ASSEMBLY.json",{"frozen_clip_source":{str(f0path.relative_to(ROOT)):f0pin},"long_rows":len(long),"wide_rows":len(wide),"fixed_contrasts":len(contrast),"cross_stage_rows":len(stage),"new_fits_rankings_or_bootstrap":False})
    print("Complete F1/P1 tables and descriptive cross-stage comparison assembled")


if __name__=="__main__":main()
