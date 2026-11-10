"""Offline annotation kit for one batch: video snippets + a one-key judging page.

For each item of the batch: the clip's 720p video from start - PAD to end + PAD, re-encoded
small, with the subtitle band (75-90% of the height, burned-in Chinese subtitles) blacked out so the judgement rests on
the Paiwan speech and the scene, not on the translation. Files are named only by position.
index.html runs from the folder with no network: keys 1 / 2 / 3 answer, R replays, B goes back;
answers autosave in the browser and "下載結果" writes a CSV (item_id, answer, seconds, time).
The kit holds no utterance id, transcript or subtitle text.
"""
import argparse, json, subprocess
from pathlib import Path

PAD, MASK_TOP, MASK_H = 2.0, 0.75, 0.15   # subtitle band on 720p TITV: about 79-86% of height

PAGE = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>排灣語語句判斷</title>
<style>
body{font-family:system-ui,"Noto Sans TC",sans-serif;background:#111;color:#eee;margin:0;padding:16px}
.wrap{max-width:860px;margin:auto} video{width:100%;background:#000;border-radius:6px}
.bar{display:flex;justify-content:space-between;margin:10px 0;font-size:15px;color:#aaa}
.btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px}
button{font-size:18px;padding:14px;border-radius:8px;border:1px solid #555;background:#222;color:#eee;cursor:pointer}
button.on{background:#2d6cdf;border-color:#2d6cdf} .small{font-size:14px;padding:8px 12px}
#intro{line-height:1.7} kbd{background:#333;padding:1px 6px;border-radius:4px}
</style></head><body><div class="wrap">
<div id="intro"><h2>排灣語語句判斷（第 BATCH 批）</h2>
<p>每段影片中間有一句排灣語（前後各多 2 秒）。請判斷<b>中間這一句</b>是不是<b>指令</b>：
說話者要聽話者（或包括說話者自己在內的大家）做某件事或不要做某件事，例如命令、要求、勸說、邀請、提醒、禁止。</p>
<p>按鍵：<kbd>1</kbd> 是指令　<kbd>2</kbd> 不是指令　<kbd>3</kbd> 無法判斷（聽不清楚、不是排灣語等）。
<kbd>R</kbd> 重播，<kbd>B</kbd> 回上一句。答案會自動儲存，可以隨時關掉再回來。全部完成後按「下載結果」，把檔案傳給古貿昌。</p>
<p>畫面下方字幕已遮住，請以聽到的排灣語為準。共 N_ITEMS 句，約 MINUTES 分鐘。</p>
<button onclick="start()">開始</button></div>
<div id="main" style="display:none">
<div class="bar"><span id="pos"></span><span id="done"></span></div>
<video id="v" playsinline controls></video>
<div class="btns"><button id="a1" onclick="ans(1)">1　是指令</button><button id="a2" onclick="ans(2)">2　不是指令</button><button id="a3" onclick="ans(3)">3　無法判斷</button></div>
<div class="bar"><button class="small" onclick="go(i-1)">B　上一句</button><button class="small" onclick="v.currentTime=0;v.play()">R　重播</button><button class="small" onclick="dl()">下載結果</button></div>
</div></div>
<script>
const ITEMS=ITEMS_JSON, KEY="pwn_kit_BATCH";
let A={};try{A=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0,t0=0;const v=document.getElementById("v");
function save(){try{localStorage.setItem(KEY,JSON.stringify(A))}catch(e){}}
function start(){document.getElementById("intro").style.display="none";document.getElementById("main").style.display="block";
 const f=ITEMS.findIndex(x=>!A[x]);go(f<0?0:f)}
function go(k){if(k<0||k>=ITEMS.length)return;i=k;v.src="clips/"+ITEMS[i]+".mp4";v.play();t0=Date.now();draw()}
function draw(){document.getElementById("pos").textContent=`第 ${i+1} / ${ITEMS.length} 句`;
 document.getElementById("done").textContent=`已完成 ${Object.keys(A).length}`;
 for(const n of[1,2,3])document.getElementById("a"+n).classList.toggle("on",A[ITEMS[i]]&&A[ITEMS[i]].a==n)}
function ans(n){A[ITEMS[i]]={a:n,s:(Date.now()-t0)/1000,t:new Date().toISOString()};save();
 if(i<ITEMS.length-1)go(i+1);else{draw();alert("全部完成，請按「下載結果」。")}}
function dl(){let c="item_id,answer,seconds,time\n";for(const k of ITEMS)if(A[k])c+=`${k},${A[k].a},${A[k].s.toFixed(1)},${A[k].t}\n`;
 const b=new Blob(["﻿"+c],{type:"text/csv"}),u=URL.createObjectURL(b),e=document.createElement("a");
 e.href=u;e.download="results_batch_BATCH.csv";e.click()}
document.addEventListener("keydown",e=>{if(document.getElementById("main").style.display=="none")return;
 if(e.key=="1"||e.key=="2"||e.key=="3")ans(+e.key);else if(e.key=="r"||e.key=="R"){v.currentTime=0;v.play()}
 else if(e.key=="b"||e.key=="B")go(i-1)});
</script></body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--videos", required=True, nargs="+", help="dirs with <clip>.mp4 (720p)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [r for r in json.load(open(a.items, encoding="utf-8"))["items"] if r["batch"] == a.batch]
    items.sort(key=lambda r: r["position"])
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    for r in items:
        src = next((Path(d) / f"{r['clip']}.mp4" for d in a.videos if (Path(d) / f"{r['clip']}.mp4").exists()), None)
        if src is None:
            raise SystemExit(f"video missing for item {r['item_id']}")
        s = max(0.0, r["start_s"] - PAD); dur = r["end_s"] + PAD - s
        dst = out / "clips" / f"{r['item_id']}.mp4"
        if dst.exists():
            continue
        vf = f"drawbox=x=0:y=ih*{MASK_TOP}:w=iw:h=ih*{MASK_H}:color=black:t=fill,scale=640:-2"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
                        "-vf", vf, "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
                        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dst)], check=True)
    ids = [r["item_id"] for r in items]
    minutes = max(1, round(len(ids) * 12 / 60))
    page = (PAGE.replace("ITEMS_JSON", json.dumps(ids)).replace("BATCH", str(a.batch))
                .replace("N_ITEMS", str(len(ids))).replace("MINUTES", str(minutes)))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"batch": a.batch, "items": len(ids), "clips": len(list((out / "clips").glob("*.mp4")))}))

if __name__ == "__main__":
    main()
