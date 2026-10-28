"""Exact blockwise Jaccard/Hamming distances, qualified against skill helper.

GPU computes metadata distances, never model fitting or label-based tuning.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import json,gzip,time,hashlib,importlib.util,collections
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
FIELDS=[('model_architecture',['model_family','architecture_stages']),('representations',['representations']),
    ('features',['features']),('preprocessing',['preprocessing_stages']),
    ('training_recipe',['training_objective','training_parameters']),('inference_composition',['inference_stages'])]
def main():
    start=time.time();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available(),'This admitted job requires its assigned GPU'
    schema_path=ROOT/'descriptors/schema-legacy-v1.json';schema=json.loads(schema_path.read_text())
    spec=importlib.util.spec_from_file_location('search_helper',ROOT/'source/search_helper_frozen.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    evaluator=helper.Schema(schema,{'sha256':hashlib.sha256(schema_path.read_bytes()).hexdigest()})
    rows=json.load(gzip.open(ROOT/'descriptors/legacy_resolved.json.gz','rt'))
    unique={}
    for row in rows:unique.setdefault(row['design_id'],row)
    designs=[unique[key] for key in sorted(unique)];n=len(designs)
    values=[evaluator.canonical(d['design']) for d in designs]
    assert evaluator.separation(values[0],values[0])[0]==0
    a=dict(values[0]);a['features']=list(a['features'])+['qualification_only_extra_feature']
    expected=(1/(len(values[0]['features'])+1))/6
    assert abs(evaluator.separation(values[0],a)[0]-expected)<1e-12
    assert evaluator.separation(values[0],a)[0]==evaluator.separation(a,values[0])[0]
    encoded={}
    for _,fields in FIELDS:
        for field in fields:
            if field in ['model_family','training_objective']:
                vocab={v:i for i,v in enumerate(sorted({v[field] for v in values}))}
                encoded[field]=torch.tensor([vocab[v[field]] for v in values],device='cuda')
            else:
                vocab={v:i for i,v in enumerate(sorted({x for v in values for x in v[field]}))}
                matrix=np.zeros((n,len(vocab)),dtype=np.float32)
                for i,value in enumerate(values):
                    for token in value[field]:matrix[i,vocab[token]]=1
                matrix=torch.from_numpy(matrix).to('cuda');encoded[field]=(matrix,matrix.sum(1))
    def distance_block(lo,hi):
        total=torch.zeros((hi-lo,n),device='cuda')
        for _,fields in FIELDS:
            group=torch.zeros_like(total)
            for field in fields:
                item=encoded[field]
                if isinstance(item,tuple):
                    mat,size=item;intersection=mat[lo:hi]@mat.T
                    union=size[lo:hi,None]+size[None,:]-intersection
                    term=torch.where(union>0,1-intersection/union,torch.zeros_like(union))
                else:term=(item[lo:hi,None]!=item[None,:]).float()
                group+=term/len(fields)
            total+=group/len(FIELDS)
        return total.cpu().numpy()
    # Qualify 96 actual archive pairs against the exact official helper. This
    # verifies distance mechanics; it is not scientific score validation.
    probes=distance_block(0,min(3,n));errors=[]
    for i in range(min(3,n)):
        for j in np.linspace(0,n-1,32,dtype=int):
            exact=evaluator.separation(values[i],values[j])[0];errors.append(abs(float(probes[i,j])-exact))
    assert max(errors)<2e-7,max(errors)
    qualification=dict(status='passed',pairs=len(errors),max_absolute_error=max(errors),
        identical_distance=0,symmetry=True,single_feature_known_case=True,
        helper_sha256=hashlib.sha256((ROOT/'source/search_helper_frozen.py').read_bytes()).hexdigest(),
        scope='Schema.canonical and Schema.separation only; rank/select controller remain unqualified.',
        device=torch.cuda.get_device_name(0),tf32=False)
    (ROOT/'validation/distance_qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    output=ROOT/'descriptors/legacy_novelty.jsonl'
    assert not output.exists(),'Preserve existing result; do not overwrite'
    with output.open('w',buffering=1) as file:
        for lo in range(0,n,96):
            hi=min(n,lo+96);dist=distance_block(lo,hi)
            for local,i in enumerate(range(lo,hi)):
                dist[local,i]=np.inf
                ix=np.argsort(dist[local],kind='stable')[:5]
                result=dict(design_id=designs[i]['design_id'],example_record_id=designs[i]['id'],
                    novelty=float(dist[local,ix].mean()),minimum_distance=float(dist[local,ix[0]]),
                    nearest=[dict(design_id=designs[j]['design_id'],distance=float(dist[local,j])) for j in ix],
                    rank_eligible=False)
                file.write(json.dumps(result)+'\n')
            (ROOT/'descriptors/novelty_status.json').write_text(json.dumps(dict(status='running',done=hi,total=n,
                seconds=time.time()-start,peak_vram_bytes=torch.cuda.max_memory_allocated()))+'\n')
    (ROOT/'descriptors/novelty_status.json').write_text(json.dumps(dict(status='complete',done=n,total=n,
        seconds=time.time()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),
        cohort='legacy56k_threefold',all_attempted_legacy_designs_included=True,
        other_cohorts_not_ranked=True,training=False))+'\n')
if __name__=='__main__':main()
