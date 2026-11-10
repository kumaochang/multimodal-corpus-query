"""Local saved-ranking validation and fixed descriptive string analysis only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import unicodedata
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NUMERIC = ROOT/'data/study3_stats/formal_release/v1'
RUNS = ROOT/'data/study3_runs'
CACHE = ROOT/'data/study3_stats/ch5_subtitle_overlap'
PACKAGE = ROOT/'data/study3_stats/ch5_subtitle_overlap/HA_W3_subtitle_fullblocks_ULFR-v1.zip'
INDEX_SHA = '9dbe943dbfe6f972a8fc35cde35ea5edd811ad91d42a54ca4b8bf78c8ecb5889'
ZIP_SHA = 'b957bdb60b838346226b315466394ed83b0b41351ff905cdc372138006413209'
ROUTES = ['xlmr_sub','tfidf_sub']
LABELS = ['Explaining','Reporting','Urging','Celebrating','Comforting']
SOURCES = ['ami','tay','pwn','tao','trv','tsu']
NORMALIZATIONS = ['NFC_WS','NFC_WS_noP']
INPUTS = {}

def require(test, message):
    if not test:
        raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def record(path, expected=None):
    p=Path(path); h=sha(p)
    require(expected is None or h==expected,'Input hash mismatch: '+p.name)
    INPUTS[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':h}
    return p

def read_json(path, expected=None):
    return json.loads(record(path,expected).read_text(encoding='utf-8'))

def write(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def npz(data):
    with np.load(data,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def validate_segment(rows,scores,expected,bank):
    require(len(rows)==len(expected) and len(set(map(int,rows)))==len(rows),'Candidate length/uniqueness')
    require(set(map(int,rows))==set(map(int,expected)),'Candidate eligibility/permutation differs')
    require(np.isfinite(scores).all() and np.all(scores[1:]<=scores[:-1]),'Nonfinite/non-descending scores')
    position={int(r):i for i,r in enumerate(bank)}
    tie=scores[1:]==scores[:-1]
    require(all(position[int(rows[i])]<position[int(rows[i+1])] for i in np.flatnonzero(tie)),
            'Equal-score ranks differ from stable bank order')
    return int(tie.sum())

def validate_pair(qx,bx,qt,bt):
    require(np.array_equal(qx,qt) and np.array_equal(bx,bt),'Unpaired route query/bank order')

def load_subtitles(clips, manifest):
    result={}; duplicates=0
    for clip in sorted(set(map(str,clips))):
        name=clip+'.subtitles.json'; path=RUNS/'subtitles'/name
        require(name in manifest['per_file_sha256']['subtitles'],'Subtitle not in source manifest')
        body=read_json(path,manifest['per_file_sha256']['subtitles'][name])
        for row in body['utterances']:
            uid=row['utt_id']; value=row.get('subtitle_text') or ''
            require(isinstance(value,str),'Assigned subtitle is not text')
            if uid in result: duplicates+=1
            require(uid not in result,'Duplicate assigned subtitle ID')
            result[uid]=value
    return result,duplicates

def read_saved():
    HERE.mkdir(exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    require(not (CACHE/'reader_cache.npz').exists(),'Existing reader cache preserved; use verify/reproduce commands')
    index=read_json(NUMERIC/'index/condition_index.json',INDEX_SHA)
    package_manifest=read_json(NUMERIC/'MANIFEST.json')
    manifest_files={x['path']:x for x in package_manifest['files']}
    units=npz(record(NUMERIC/'index/unit_rows.npz',index['unit_rows_sha256']))
    with np.load(record(RUNS/'features_full/features.npz','80ab8584dd41383226e0986055480ead9cac774b9e3b5928d4f2bf45d80d1704'),allow_pickle=False) as z:
        ids=z['ids'].astype(str)
    require(np.array_equal(ids,units['unit_id']) and len(set(ids))==len(ids),'Global feature/unit ID order differs')
    source_manifest=read_json(ROOT/'data/study3_runs/index_manifest.json')
    rank_source=record(ROOT/'analysis/study3/formal_release/fr_rank_supplement.py',
                       'dd5bd1048b53faab2f98b75bb6e2346421384d34f9dc7557171ecd733bc7642d')
    require("assert S.dtype == np.float32" in rank_source.read_text(encoding='utf-8') and
            "np.argsort(-S[row, ci], kind='stable')" in rank_source.read_text(encoding='utf-8'),
            'Original pre-ranking score precision/ordering contract not verified')
    require(source_manifest['inputs']['data/study3_runs/features_full/features.npz']['sha256']==INPUTS['data/study3_runs/features_full/features.npz']['sha256'],'Subtitle provenance uses different feature IDs')
    model=npz(record(NUMERIC/'supplement/runA/execution/HA_W3_model_data.npz','6417e9783238a7af27baecf95817e095a2998d6ee5e26d337daf4d9d5dde1429'))
    q=model['all_q'][::6].astype(int)
    require(len(q)==2753 and len(set(q))==2753 and np.array_equal(np.repeat(q,6),model['all_q']),'HA query pairing/order differs')
    require(np.array_equal(model['all_query'],np.repeat(np.arange(len(q)),6)),'HA query index differs')
    lid=units['lid_answer'][q]; mf=units['mf_class'][q]
    speech=np.where(lid=='mandarin',0,np.where((lid=='not_chinese')&(mf!='mandarin'),1,-1))
    require(np.all(speech>=0),'Not the HA M-union-I population')
    positions={int(r):i for i,r in enumerate(q)}
    neighbors=np.full((len(q),2,3),-1,dtype=np.int32); scores=np.full((len(q),2,3),np.nan,dtype=np.float32)
    counts=np.zeros((len(q),2),dtype=np.int32); verified=[]; per_route={}; total_entries=0
    record(PACKAGE,ZIP_SHA)
    with zipfile.ZipFile(PACKAGE) as archive:
        names=archive.namelist(); require(len(names)==len(set(names))==13,'ZIP members differ')
        for member in archive.infolist():
            rel=PurePosixPath(member.filename)
            require(not rel.is_absolute() and '..' not in rel.parts and not stat.S_ISLNK(member.external_attr>>16),'Unsafe ZIP entry')
        delivery=json.loads(archive.read('MANIFEST.json'))
        require(delivery['release']==index['release'] and delivery['source_index_sha256']==INDEX_SHA,'Delivery identity differs')
        rows={r['condition']:r for r in delivery['files']}
        expected={k:v for k,v in index['conditions'].items() if k.startswith('supplement_new|A|six_common|union_MI|W3|') and v['representation'] in ROUTES}
        require(set(rows)==set(expected) and len(rows)==12,'Wrong block condition set')
        for cid,e in sorted(expected.items()):
            payload=archive.read(rows[cid]['path'])
            require(hashlib.sha256(payload).hexdigest()==e['sha256']==rows[cid]['sha256'],'Block hash differs')
            require(rows[cid]['path']=='rankings/'+e['file'] and len(payload)==rows[cid]['bytes'],'Block path/size differs')
            require(e['rule']=='registered_float32_cosine_stable_desc:exclude_self_and_same_day','Wrong ranking rule')
            block=npz(io.BytesIO(payload)); path=NUMERIC/'rankings'/e['topk_file']
            require(manifest_files['rankings/'+e['topk_file']]['sha256']==e['topk_sha256'],'Top10 index/manifest disagreement')
            view=npz(record(path,e['topk_sha256']))
            qr=block['query_rows']; bank=block['bank_rows']; off=block['offsets']; rr=block['rank_rows']; sc=block['rank_scores']
            require(sc.dtype==np.dtype('<f4') and np.array_equal(qr,view['query_rows']),'Block/view query order or dtype')
            require(len(qr)==e['queries'] and len(rr)==e['entries'] and off[0]==0 and off[-1]==len(rr) and np.all(np.diff(off)>=0),'Offsets/entry count')
            require(np.all((bank>=0)&(bank<len(ids))) and len(set(map(int,bank)))==len(bank),'Invalid bank IDs')
            wanted=q[units['lang'][q]==e['source']]
            require(set(map(int,bank))==set(map(int,wanted))==set(map(int,qr)),'Bank/query not HA source population')
            ri=ROUTES.index(e['representation']); per_route[(ri,e['source'])]=(qr.copy(),bank.copy())
            ties=0
            for k,query in enumerate(qr):
                require(int(query) in positions,'Unknown query ID')
                rank=rr[off[k]:off[k+1]]; score=sc[off[k]:off[k+1]]
                allowed=bank[(bank!=query)&(units['day'][bank]!=units['day'][query])]
                ties+=validate_segment(rank,score,allowed,bank)
                require(len(rank)==view['n_cand'][k],'Candidate count differs from top10')
                top=min(10,len(rank)); require(np.array_equal(rank[:top],view['top_rows'][k,:top]),'Full/top10 IDs differ')
                require(np.array_equal(score[:top].view(np.uint32),view['top_scores'][k,:top].view(np.uint32)),'Full/top10 score bits differ')
                pos=positions[int(query)]; n=min(3,len(rank))
                require(counts[pos,ri]==0,'Duplicate query condition')
                counts[pos,ri]=len(rank);neighbors[pos,ri,:n]=rank[:n];scores[pos,ri,:n]=score[:n]
                hit=float(units['haiku'][query]==units['haiku'][rank[0]])
                require(hit==model['all_hit'][6*pos+[3,4][ri]],'Saved model P@1 hit differs')
            total_entries+=len(rr);verified.append({'condition':cid,'queries':len(qr),'entries':len(rr),'equal_adjacent_score_pairs':ties})
    for source in SOURCES:
        validate_pair(*per_route[(0,source)],*per_route[(1,source)])
    require(np.all(counts>0) and np.array_equal(counts[:,0],counts[:,1]),'Unpaired/empty candidates')
    texts,duplicates=load_subtitles(units['clip'][q],source_manifest)
    required=set(map(int,q))|set(map(int,neighbors[neighbors>=0]))
    require(all(str(ids[r]) in texts for r in required),'Missing exact subtitle mapping')
    empty=sum(not texts[str(ids[r])].strip() for r in required)
    source=np.array([SOURCES.index(x) for x in units['lang'][q]],np.int8)
    label=np.array([LABELS.index(x) for x in units['haiku'][q]],np.int8)
    days=sorted(set(zip(source.tolist(),units['day'][q].tolist())))
    day=np.array([days.index((int(s),d)) for s,d in zip(source,units['day'][q])],np.int16)
    clips=sorted(set(units['clip'][q].tolist()));clip=np.array([clips.index(x) for x in units['clip'][q]],np.int16)
    hits=units['haiku'][q,None]==units['haiku'][neighbors[:,:,0]]
    np.savez_compressed(CACHE/'reader_cache.npz',query_rows=q,ids=ids[q],neighbors=neighbors,scores=scores,n_candidates=counts,
                        source=source,label=label,speech=speech.astype(np.int8),day=day,clip=clip,hit=hits)
    audit={'passed':True,'UTC':datetime.now(timezone.utc).isoformat(),'release':index['release'],'reader':'independent ZIP reader; original reader and manifests unchanged',
           'queries':len(q),'source_days':len(days),'clips':len(clips),'conditions':verified,'full_entries_checked':total_entries,
           'top10_bitwise_checked':True,'ordered_route_pairing_checked':True,'model_saved_hits_checked':True,
           'tie_precision_provenance':'Pinned fr_rank_supplement.score asserts float32 S before stable argsort; compares stored candidate-bank positions, not global row order',
           'subtitle_required_units':len(required),'subtitle_files':len(set(units['clip'][q])),'subtitle_duplicate_IDs':duplicates,'subtitle_missing_IDs':0,'required_empty_or_whitespace_subtitles':empty,
           'labels':LABELS,'sources':SOURCES,'routes':ROUTES,'inputs':INPUTS,'reader_cache_sha256':sha(CACHE/'reader_cache.npz')}
    write(HERE/'READER_AUDIT.json',audit)
    print(json.dumps({k:audit[k] for k in ('passed','queries','source_days','clips','full_entries_checked','subtitle_missing_IDs','required_empty_or_whitespace_subtitles')}))

def normalize(text, no_punctuation=False):
    s=' '.join(unicodedata.normalize('NFC',text).split())
    if no_punctuation: s=' '.join(''.join(c for c in s if not unicodedata.category(c).startswith('P')).split())
    return s

def dice(a,b):
    if min(len(a),len(b))<2:return float('nan')
    ca=Counter(zip(a,a[1:]));cb=Counter(zip(b,b[1:]))
    return 2*sum((ca&cb).values())/(len(a)+len(b)-2)

def route_average(values,k):
    return float(np.mean(values[:k])) if k and np.isfinite(values[:k]).all() else float('nan')

def paired_valid(values):
    return np.isfinite(values).all(axis=-1)

def describe(a):
    a=np.asarray(a,float); require(np.isfinite(a).all(),'Nonfinite distribution input')
    if not len(a):return dict(n=0,mean=None,sd=None,min=None,q25=None,median=None,q75=None,max=None)
    q=np.quantile(a,[.25,.5,.75],method='linear')
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else None,
                min=float(a.min()),q25=float(q[0]),median=float(q[1]),q75=float(q[2]),max=float(a.max()))

def groups(cache):
    for label in [-1]+list(range(5)):
        for speech in [-1,0,1]:
            mask=np.ones(len(cache['ids']),bool)
            if label>=0:mask&=cache['label']==label
            if speech>=0:mask&=cache['speech']==speech
            yield ('all' if label<0 else LABELS[label]),('all' if speech<0 else ['M','I'][speech]),mask

def support(c,mask):
    return {'n':int(mask.sum()),'days':len(set(c['day'][mask])),'clips':len(set(c['clip'][mask])),'sources':len(set(c['source'][mask]))}

def freeze():
    audit=read_json(HERE/'READER_AUDIT.json');require(audit['passed'],'Reader audit incomplete')
    require(sha(CACHE/'reader_cache.npz')==audit['reader_cache_sha256'],'Reader cache changed')
    files=[HERE/'PLAN.md',HERE/'pipeline.py',HERE/'READER_AUDIT.json']
    require(not (HERE/'PLAN_FREEZE.json').exists(),'Original plan freeze preserved')
    write(HERE/'PLAN_FREEZE.json',{'UTC':datetime.now(timezone.utc).isoformat(),'new_overlap_results_seen':False,
         'files':{p.name:sha(p) for p in files},'reader_cache_sha256':audit['reader_cache_sha256'],'inputs':audit['inputs'],
         'environment':{'python':platform.python_version(),'numpy':np.__version__,'Unicode':unicodedata.unidata_version,'platform':platform.platform()}})
    print('Plan/code/input freeze saved before overlap computation.')

def analyze(reproduce=False):
    frozen=read_json(HERE/'PLAN_FREEZE.json')
    for name,h in frozen['files'].items():require(sha(HERE/name)==h,'Frozen plan/code/audit changed: '+name)
    for name,x in frozen['inputs'].items():require(sha(ROOT/name)==x['sha256'],'Input changed: '+name)
    require(sha(CACHE/'reader_cache.npz')==frozen['reader_cache_sha256'],'Reader cache changed')
    c=npz(CACHE/'reader_cache.npz');n=len(c['ids'])
    with np.load(NUMERIC/'index/unit_rows.npz',allow_pickle=False) as z:
        ids=z['unit_id'];clips=z['clip']
    manifest=read_json(ROOT/'data/study3_runs/index_manifest.json');texts,_=load_subtitles(clips[c['query_rows']],manifest)
    ds=np.full((2,n,2,3),np.nan);ex=ds.copy();qlen=np.zeros((2,n),int);nlen=np.full((2,n,2,3),-1,int)
    for ni in range(2):
        for i,row in enumerate(c['query_rows']):
            q=normalize(texts[str(ids[row])],bool(ni));qlen[ni,i]=len(q)
            for ri in range(2):
                for k,r in enumerate(c['neighbors'][i,ri]):
                    if r<0:continue
                    t=normalize(texts[str(ids[r])],bool(ni));nlen[ni,i,ri,k]=len(t)
                    ds[ni,i,ri,k]=dice(q,t)
                    if min(len(q),len(t))>=2:ex[ni,i,ri,k]=float(q==t)
    values=np.full((2,2,n,2),np.nan);exact=values.copy();length=values.copy()
    for ni in range(2):
        for wi,window in enumerate([1,3]):
            for i in range(n):
                for ri in range(2):
                    k=min(window,int(c['n_candidates'][i,ri]))
                    values[ni,wi,i,ri]=route_average(ds[ni,i,ri],k)
                    exact[ni,wi,i,ri]=route_average(ex[ni,i,ri],k)
                    length[ni,wi,i,ri]=float(nlen[ni,i,ri,:k].mean())
    valid=paired_valid(values);state=c['hit'][:,0].astype(int)+2*c['hit'][:,1].astype(int)
    state_names=['neither','xlmr_only','tfidf_only','both']
    rows=[]
    def summarize(ni,wi,mask):
        a=values[ni,wi,mask];e=exact[ni,wi,mask]
        return {'support':support(c,mask),'dice_by_route':[describe(a[:,r]) for r in range(2)],
                'paired_dice_TFIDF_minus_XLMR':describe(a[:,1]-a[:,0]),
                'exact_by_route':[{'sum_query_neighbor_fractions':float(e[:,r].sum()),'denominator_queries':len(e),'mean':float(e[:,r].mean()) if len(e) else None} for r in range(2)],
                'paired_exact_TFIDF_minus_XLMR':describe(e[:,1]-e[:,0]),'query_length':describe(qlen[ni,mask]),
                'neighbor_mean_length_by_route':[describe(length[ni,wi,mask,r]) for r in range(2)]}
    for ni,norm in enumerate(NORMALIZATIONS):
        for wi,window in enumerate([1,3]):
            for label,speech,g in groups(c):
                m=g&valid[ni,wi];native=np.isfinite(values[ni,wi])
                reasons=[]
                for ri in range(2):
                    empty=qlen[ni]==0;short=qlen[ni]==1
                    for i in range(n):
                        k=min(window,int(c['n_candidates'][i,ri]));empty[i]|=np.any(nlen[ni,i,ri,:k]==0);short[i]|=np.any(nlen[ni,i,ri,:k]==1)
                    reasons.append({'query_empty':int(np.sum(g&(qlen[ni]==0))),'query_short':int(np.sum(g&(qlen[ni]==1))),
                                    'any_required_empty':int(np.sum(g&empty)),'any_required_short':int(np.sum(g&short))})
                rows.append(dict(normalization=norm,rank_window=window,model_class=label,speech_group=speech,
                    full_support=support(c,g),native_valid_by_route=[int(np.sum(g&native[:,r])) for r in range(2)],
                    excluded_from_paired=int(np.sum(g&~valid[ni,wi])),exclusion_reasons_by_route=reasons,
                    available_neighbors_histogram=dict(Counter(str(min(window,int(k))) for k in c['n_candidates'][g,0])),
                    **summarize(ni,wi,m),match_states=[dict(state=state_names[s],full_count=int(np.sum(g&(state==s))),
                        **summarize(ni,wi,m&(state==s))) for s in range(4)]))
    common=[]
    for label,speech,g in groups(c):
        for ni in range(2):
            m=g&valid[ni,0]&valid[ni,1]
            common.append(dict(kind='rank_windows_common_queries',normalization=NORMALIZATIONS[ni],model_class=label,speech_group=speech,
                               rank1=summarize(ni,0,m),top3=summarize(ni,1,m)))
        for wi in range(2):
            m=g&valid[0,wi]&valid[1,wi]
            common.append(dict(kind='normalizations_common_queries',rank_window=[1,3][wi],model_class=label,speech_group=speech,
                               primary=summarize(0,wi,m),no_punctuation=summarize(1,wi,m)))
    coverage=[dict(source=SOURCES[s],model_class=label,speech_group=sp,**support(c,g&(c['source']==s))) for s in range(6) for label,sp,g in groups(c)]
    result={'schema':'ha-w3-subtitle-overlap/1','plan_sha256':frozen['files']['PLAN.md'],'routes':ROUTES,'normalizations':NORMALIZATIONS,
            'difference_direction':'TF-IDF minus XLM-R','descriptive_only':True,'rows':rows,'common_support_checks':common,'source_coverage':coverage}
    output=(CACHE/'summary_reproduced.json') if reproduce else (HERE/'SUMMARY.json')
    if not reproduce:require(not output.exists(),'Original results preserved; use reproduce')
    write(output,result)
    if not reproduce:
        np.savez_compressed(CACHE/'derived_cache.npz',dice_neighbors=ds,exact_neighbors=ex,query_length=qlen,neighbor_length=nlen,
                            values=values,exact=exact,valid=valid,match_state=state)
    else:require(output.read_bytes()==(HERE/'SUMMARY.json').read_bytes(),'Description not byte reproducible')
    print(json.dumps({'descriptive_rows':len(rows),'common_support_rows':len(common),'source_support_rows':len(coverage),'reproduced':reproduce}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['reader','freeze','analyze','reproduce']);args=parser.parse_args()
    if args.command=='reader':read_saved()
    elif args.command=='freeze':freeze()
    else:analyze(args.command=='reproduce')
