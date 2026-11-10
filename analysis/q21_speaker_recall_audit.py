"""Audit saved ari phone output and gold counts. Standard library only.
No recognizer/model imports, audio access, consultant files, network or resampling.
"""
import argparse,collections,hashlib,json,re,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
WINDOWS={'unwindowed':None,'first25':25,'first15':15,'first10':10}
XMLS = {}  # Provided by --recording-map.

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ratio(n,d):return {'numerator':n,'denominator':d,'value':n/d if d else None}
def unique(rows,key):
 out={}
 for r in rows:
  if r[key] in out:raise ValueError('Duplicate '+key)
  out[r[key]]=r
 return out

def stats(rows,window):
 c=collections.Counter(r['windows'][window]['confusion'] for r in rows)
 out={k:c[k] for k in ('TP','FN','FP','TN')}
 out.update({'clips':len(rows),'gold_tokens':sum(r['gold_ari_tokens'] for r in rows),
   'gold_positive_clips':c['TP']+c['FN'],'predicted_positive_clips':c['TP']+c['FP'],
   'detected_occurrences':sum(r['windows'][window]['detected_occurrences'] for r in rows),
   'precision':ratio(c['TP'],c['TP']+c['FP']),'recall':ratio(c['TP'],c['TP']+c['FN'])})
 assert sum(c.values())==len(rows)
 return out

def old_tables(reference):
 overall=reference['overall'];recordings={};locators={}
 for src in XMLS:
  key=hashlib.sha256(src.encode('utf-8')).hexdigest()
  value=reference['recordings'][key]
  recordings[src]=dict(value,locator='analysis/phone_detection_reference.json#/recordings/'+key)
 for w in WINDOWS:locators[w]='analysis/phone_detection_reference.json#/overall/'+w
 assert set(overall)==set(WINDOWS) and set(recordings)==set(XMLS)
 return overall,recordings,locators

def self_test():
 assert ratio(0,0)['value'] is None and ratio(0,2)['value']==0
 # Count pooling, never an unweighted mean of recording recalls.
 assert ratio(6+0,7+1)['value']==.75 and (6/7+0)/2!=.75
 assert len(re.findall('ari','ariari'))==2 and bool(re.search('ari','ariari')) is True
 return ['undefined denominator versus zero numerator','pool counts before ratios','occurrences versus clip hit']

def run(root,out_prefix):
 names=['analysis/phone_detection_reference.json','pipeline/qc/results/armA_q21_reproduction.json',
        'pipeline/analysis/armA.py','pipeline/qc/phone_layer_qc.py',
        'data/ntu_paiwan_asr/clips.json','data/ntu_paiwan_asr/gold_index.json',
        'data/qc/phone_qc.jsonl','data/qc/manifest.json','analysis/q21_speaker_recall_audit.py']
 names += ['data/ntu_paiwan_asr/xml/'+x for x in XMLS.values()]
 before={n:{'sha256_before':digest(root/n),'bytes':(root/n).stat().st_size,
            'role':'protected computation/source'} for n in names}
 clips=json.loads((root/'data/ntu_paiwan_asr/clips.json').read_text(encoding='utf-8'))
 gold=json.loads((root/'data/ntu_paiwan_asr/gold_index.json').read_text(encoding='utf-8'))
 manifest=[r for r in json.loads((root/'data/qc/manifest.json').read_text()) if r['set']=='ntu']
 phones=[]
 for line_no,line in enumerate((root/'data/qc/phone_qc.jsonl').read_text().splitlines(),1):
  r=json.loads(line)
  if r['set']=='ntu':phones.append(dict(r,line_no=line_no))
 byphone=unique(phones,'id');bymanifest=unique(manifest,'id')
 byclip=unique([dict(r,clip_key=Path(r['clip']).name,index=i) for i,r in enumerate(clips)],'clip_key')
 assert len(byclip)==102 and set(byclip)==set(gold)==set(byphone)==set(bymanifest)
 assert set(r['src'] for r in clips)==set(XMLS)
 xmlmap={};speakers={}
 for src,name in XMLS.items():
  tree=ET.parse(root/'data/ntu_paiwan_asr/xml'/name).getroot()
  assert tree.attrib['audio']==src
  speaker=re.fullmatch(r'NTU Paiwan Data, Participant ([^,]+), file .+',tree.attrib['source']).group(1)
  assert name.startswith(speaker+'__')
  speakers[src]=speaker;xmlmap[src]={s.attrib['id']:s for s in tree.findall('S')}
 rows=[]
 for key,clip in byclip.items():
  assert gold[key]==[clip['ari'],clip['maya'],clip['src']]
  sid=Path(key).stem.removeprefix(Path(clip['src']).stem+'_');s=xmlmap[clip['src']][sid]
  # Both FORM tiers match the saved gold index in these selected 102 units.
  forms=s.findall('FORM');assert {f.attrib['kindOf'] for f in forms}=={'original','standard'}
  assert all((f.text or '')==clip['form'] for f in forms)
  assert all(len(re.findall(r'\bari\b',f.text or '',re.I))==clip['ari'] for f in forms)
  audio=s.find('AUDIO');assert audio.attrib['file']==key
  assert float(audio.attrib['start'])==clip['start'] and float(audio.attrib['end'])==clip['end']
  p=byphone[key];assert p['iso']=='pwn' and bymanifest[key]['iso']=='pwn'
  assert Path(bymanifest[key]['wav']).name==key and isinstance(p['shipped'],list)
  seq=p['shipped'];assert all(isinstance(x,str) for x in seq)
  row={'clip_id':key,'source_recording':clip['src'],'speaker':speakers[clip['src']],
    'gold_ari_tokens':clip['ari'],'gold_positive':clip['ari']>0,
    'locators':{'clip_json_pointer':'/'+str(clip['index']),'gold_index_key':key,
      'phone_jsonl_line':p['line_no'],'xml_path':'data/ntu_paiwan_asr/xml/'+XMLS[clip['src']],
      'xml_S_id':sid,'FORM_kinds':['original','standard']},'windows':{}}
  for window,n in WINDOWS.items():
   selected=seq if n is None else seq[:n]
   count=len(re.findall('ari',''.join(selected)))
   assert count==len(re.findall(r'a\s*r\s*i',' '.join(selected)))
   pred=count>0;g=row['gold_positive']
   row['windows'][window]={'detected_occurrences':count,'predicted_positive':pred,
      'confusion':'TP' if pred and g else 'FN' if g else 'FP' if pred else 'TN'}
  assert row['windows']['first15']['detected_occurrences']<=row['windows']['unwindowed']['detected_occurrences']
  rows.append(row)
 overall={w:stats(rows,w) for w in WINDOWS}
 rec={src:{w:stats([r for r in rows if r['source_recording']==src],w) for w in WINDOWS} for src in sorted(XMLS)}
 sp={s:{w:stats([r for r in rows if r['speaker']==s],w) for w in WINDOWS} for s in sorted(set(speakers.values()))}
 old,oldrec,oldloc=old_tables(json.loads((root/'analysis/phone_detection_reference.json').read_text(encoding='utf-8')))
 repro=json.loads((root/'pipeline/qc/results/armA_q21_reproduction.json').read_text())
 reconciliation={}
 for w,t in overall.items():
  key='ari' if w=='unwindowed' else 'ari_'+w
  actual=[t['detected_occurrences']]+[t[k] for k in ('TP','FN','FP','TN')]
  assert actual==repro['this_run'][key]
  assert [old[w]['detected']]+[old[w][k] for k in ('TP','FN','FP','TN')]==repro['q21_table'][key]
  assert all(t[k]==old[w][k] for k in ('TP','FN','FP','TN'))
  assert all(round(t[k]['value'],2)==old[w][k] for k in ('precision','recall'))
  reconciliation[w]={'q21_locator':oldloc[w],'reproduction_pointer':'/this_run/'+key,
    'confusion_counts_match_both_saved_summaries':True,'old_detected':old[w]['detected'],
    'recomputed_occurrences':t['detected_occurrences'],'difference':t['detected_occurrences']-old[w]['detected']}
 for src in rec:
  assert rec[src]['unwindowed']['gold_tokens']==oldrec[src]['gold_tokens']
  for w in ('unwindowed','first15'):
   text=' / '.join('n/a' if rec[src][w][k]['value'] is None else f"{rec[src][w][k]['value']:.2f}" for k in ('precision','recall'))
   assert text==oldrec[src][w]
 for w in WINDOWS:
  for k in ('clips','gold_tokens','TP','FN','FP','TN','detected_occurrences'):
   assert overall[w][k]==sum(r[w][k] for r in rec.values())==sum(r[w][k] for r in sp.values())
 assert overall['unwindowed']['gold_tokens']==33 and overall['unwindowed']['gold_positive_clips']==22
 numeric=root/'output/q21_speaker_recall_audit/clip_counts.json';numeric.parent.mkdir(parents=True,exist_ok=True)
 payload=(json.dumps({'scope':'derived numeric identifiers/counts only; no text or phones','clips':rows},indent=2,allow_nan=False)+'\n').encode()
 if numeric.exists():assert numeric.read_bytes()==payload,'Existing derived numeric cache differs; preserve and investigate'
 else:numeric.write_bytes(payload)
 for name,info in before.items():
  info['sha256_after']=digest(root/name);info['unchanged']=info['sha256_before']==info['sha256_after']
  if name in names:assert info['unchanged'],name+' changed during audit'
 result={'audit_date':'2026-10-29','initial_task_head':'65bd8b528696ead7945e9cbcdd9e6c330fb9abef',
   'head_at_calculation':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip(),
   'method':{'evaluation_unit':'sentence clip, not token','gold':'case-insensitive word-boundary ari in XML FORM; agrees with both retained FORM tiers and frozen gold indices',
      'prediction':'nonoverlapping ari regex occurrences in concatenated saved shipped/pwn phones; first N means phone-list entries, not characters',
      'precision':'TP/(TP+FP)','recall':'TP/(TP+FN)','undefined':'null when denominator is zero',
      'speaker_pooling':'sum confusion counts over all recordings belonging to a metadata-identified participant; never mean recording ratios'},
   'recording_to_speaker':speakers,'overall':overall,'by_recording':rec,'by_speaker':sp,
   'source_locations':{'by_recording':{src:{'clips_json_pointers':[r['locators']['clip_json_pointer'] for r in rows if r['source_recording']==src],
      'phone_jsonl_lines':[r['locators']['phone_jsonl_line'] for r in rows if r['source_recording']==src],
      'xml_path':'data/ntu_paiwan_asr/xml/'+XMLS[src],
      'xml_selector':'TEXT/S/FORM[@kindOf=standard]; matching AUDIO file basename',
      'numeric_cache_selector':{'source_recording':src},'q21_recording_row':oldrec[src]['locator']} for src in rec},
      'by_speaker':{s:{'recordings':[r for r in speakers if speakers[r]==s],
      'operation':'sum confusion counts, then divide','numeric_cache_selector':{'speaker':s}} for s in sp}},
   'recording_source_locators':oldrec,'reconciliation':reconciliation,
   'coverage':{'sentence_clips':102,'source_recordings':5,'all_speakers':4,'speakers_with_gold_ari':3,
       'ari_tokens':33,'ari_positive_clips':22,'ari_negative_clips':80},
   'numeric_cache':{'path':str(numeric.resolve()),'bytes':numeric.stat().st_size,'sha256':digest(numeric),
       'ordered_clip_ids_sha256':hashlib.sha256(('\n'.join(r['clip_id'] for r in rows)+'\n').encode()).hexdigest()},
   'checks':{'self_tests':self_test(),'clip_gold_xml_and_audio_time_matches':102,'exact_id_alignment':True,
      'all_window_confusion_counts_match':True,'recording_rounded_metrics_match_q21':True,
      'recording_speaker_overall_counts_reconcile':True,'protected_hashes_unchanged':True},
   'limits':['Old Q21 first15 occurrence count 29 is not reproduced; saved QC gives 30. Original Q21 per-clip detection/version archive was not located in the cited data directory or source index, so the historical one-occurrence cause and old clip-by-clip identity cannot be determined.',
      'Matching aggregate confusion counts establish unchanged reported precision/recall, not identity of old and later predicted clips.',
      'No model inference, audio decoding, consultant data, API, ranking, resampling or thesis edits.'],
   'source_hashes':before}
 out_prefix=Path(out_prefix);out_prefix.parent.mkdir(parents=True,exist_ok=True)
 out_prefix.with_suffix('.json').write_bytes((json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
 print(json.dumps({'clips':102,'recordings':5,'speakers':4,'ari_positive_speakers':3,
    'speaker_first15_recall':{s:t['first15']['recall'] for s,t in sp.items()},
    'protected_sources':len(names),'hashes_unchanged':True,'outputs':str(out_prefix)}))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,default=ROOT);p.add_argument('--out-prefix',type=Path);p.add_argument('--recording-map',type=Path,required=True)
 a=p.parse_args();root=a.repo_root.resolve();XMLS=json.loads(a.recording_map.read_text(encoding='utf-8'));run(root,a.out_prefix or root/'analysis/q21_speaker_recall_audit')
