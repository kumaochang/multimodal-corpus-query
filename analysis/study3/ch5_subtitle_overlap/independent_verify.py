"""Independent scalar recomputation of saved overlap descriptions, no inference."""
from pathlib import Path
import hashlib,json,math,re,statistics,unicodedata
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CACHE=ROOT/'data/study3_stats/ch5_subtitle_overlap'
N=ROOT/'data/study3_stats/formal_release/v1'

def check(x,y):
    if x is None or y is None:assert x==y,(x,y)
    else:assert math.isclose(float(x),float(y),rel_tol=1e-12,abs_tol=1e-12),(x,y)

def recompute_string(s,alt):
    s=unicodedata.normalize('NFC',s)
    if alt:s=''.join(ch for ch in s if unicodedata.category(ch)[0]!='P')
    return re.sub(r'\s+',' ',s).strip()

def independent_dice(a,b):
    if len(a)<2 or len(b)<2:return float('nan')
    def frequencies(s):
        d={}
        for j in range(len(s)-1):
            k=s[j:j+2];d[k]=d.get(k,0)+1
        return d
    x,y=frequencies(a),frequencies(b)
    numerator=sum(min(x[k],y.get(k,0)) for k in x)
    return (2*numerator)/(len(a)+len(b)-2)

def distribution(values,saved):
    assert len(values)==saved['n']
    if not values:
        assert all(v is None for k,v in saved.items() if k!='n');return
    check(math.fsum(values)/len(values),saved['mean'])
    check(statistics.stdev(values) if len(values)>1 else None,saved['sd'])
    v=sorted(values);check(v[0],saved['min']);check(v[-1],saved['max'])
    for q,name in [(.25,'q25'),(.5,'median'),(.75,'q75')]:
        p=(len(v)-1)*q;l=math.floor(p);h=math.ceil(p);check(v[l]+(p-l)*(v[h]-v[l]),saved[name])

def main():
    with np.load(CACHE/'reader_cache.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
    with np.load(CACHE/'derived_cache.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(N/'index/unit_rows.npz',allow_pickle=False) as z:ids=z['unit_id'];clips=z['clip']
    audit=json.loads((HERE/'READER_AUDIT.json').read_text());summary=json.loads((HERE/'SUMMARY.json').read_text())
    for name,info in audit['inputs'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==info['sha256']
    texts={}
    for clip in set(clips[c['query_rows']]):
        body=json.loads((ROOT/'data/study3_runs/subtitles'/f'{clip}.subtitles.json').read_text(encoding='utf-8'))
        for r in body['utterances']:
            assert r['utt_id'] not in texts;texts[r['utt_id']]=r.get('subtitle_text') or ''
    raw_checks=0
    for alt in range(2):
        for i,row in enumerate(c['query_rows']):
            q=recompute_string(texts[str(ids[row])],alt);assert len(q)==d['query_length'][alt,i]
            for route in range(2):
                for k,r in enumerate(c['neighbors'][i,route]):
                    if r<0:continue
                    t=recompute_string(texts[str(ids[r])],alt);assert len(t)==d['neighbor_length'][alt,i,route,k]
                    value=independent_dice(q,t);saved=d['dice_neighbors'][alt,i,route,k]
                    assert (math.isnan(value) and math.isnan(saved)) or value==saved
                    expected=float(q==t) if min(len(q),len(t))>=2 else float('nan')
                    v=d['exact_neighbors'][alt,i,route,k]
                    assert (math.isnan(expected) and math.isnan(v)) or expected==v
                    raw_checks+=1
    labels=['Explaining','Reporting','Urging','Celebrating','Comforting'];compared=0
    for row in summary['rows']:
        ni=['NFC_WS','NFC_WS_noP'].index(row['normalization']);window=row['rank_window']
        full=[i for i in range(len(c['ids'])) if (row['model_class']=='all' or labels[c['label'][i]]==row['model_class']) and
              (row['speech_group']=='all' or ['M','I'][c['speech'][i]]==row['speech_group'])]
        assert len(full)==row['full_support']['n']
        good=[];a=[[],[]];exact=[[],[]]
        for i in full:
            k=min(window,int(c['n_candidates'][i,0]));v=[d['dice_neighbors'][ni,i,r,:k].tolist() for r in range(2)]
            if not all(math.isfinite(x) for r in v for x in r):continue
            good.append(i)
            for r in range(2):
                a[r].append(math.fsum(v[r])/k);exact[r].append(math.fsum(d['exact_neighbors'][ni,i,r,:k])/k)
        assert len(good)==row['support']['n']
        assert len(set(c['day'][good]))==row['support']['days'] and len(set(c['clip'][good]))==row['support']['clips']
        for r in range(2):
            distribution(a[r],row['dice_by_route'][r]);check(math.fsum(exact[r]),row['exact_by_route'][r]['sum_query_neighbor_fractions'])
        distribution([y-x for x,y in zip(a[0],a[1])],row['paired_dice_TFIDF_minus_XLMR'])
        distribution([y-x for x,y in zip(exact[0],exact[1])],row['paired_exact_TFIDF_minus_XLMR'])
        assert sum(s['full_count'] for s in row['match_states'])==len(full)
        assert sum(s['support']['n'] for s in row['match_states'])==len(good)
        compared+=1
    result={'passed':True,'independent_neighbor_measure_checks':raw_checks,'independently_recomputed_main_group_rows':compared,
            'method':'Original assigned strings, independent scalar normalization/bigram counts; math.fsum/statistics and scalar linear quantiles',
            'no_sampling_or_pipeline':True}
    (HERE/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
