"""Predict from a retained indexed Kumo context and optional saved head/spline.

Queries must already use the retained feature recipe; this is not a raw-table
feature builder. Reconstructs the frozen context, never refits a learned head.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,time
import numpy as np
import pandas as pd
import joblib
import baseline as b
from gpu_backend import Engine
from robustness import A

parser=argparse.ArgumentParser()
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--feature-table',type=Path)
parser.add_argument('--targets',type=Path)
parser.add_argument('--checkpoint',type=Path)
query_source=parser.add_mutually_exclusive_group(required=True)
query_source.add_argument('--query',type=Path)
query_source.add_argument('--rows',type=Path)
query_source.add_argument('--test',action='store_true')
parser.add_argument('--row-key',default='validation')
parser.add_argument('--head',type=Path)
parser.add_argument('--spline',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert args.output.suffix=='.npz','Use an explicit .npz output filename'
assert not args.output.exists() and not args.output.with_suffix('.complete.json').exists()
assert bool(args.head)==bool(args.spline),'Supply both saved downstream branches, or neither'
started=time.time();canonical=joblib.load(args.model)
assert canonical['kind']=='indexed_canonical_Kumo_context'
feature_path=args.feature_table or Path(canonical['feature_table'])
target_path=args.targets or Path(canonical['target_source'])
assert b.sha(feature_path)==canonical['feature_table_sha256']
assert b.sha(target_path)==canonical['target_source_sha256']
checkpoint=args.checkpoint or A/'ranking-pipeline-20261020/assets/classifier.pt'
assert b.sha(checkpoint)==canonical['checkpoint_sha256']
frame=pd.read_parquet(feature_path)
labels=pd.read_parquet(target_path,columns=['application_id','target'])
assert frame.application_id.iloc[:len(labels)].tolist()==labels.application_id.tolist()
indices=canonical['context_indices'];columns=canonical['columns']
assert len(set(indices))==len(indices) and min(indices)>=0 and max(indices)<len(labels)
assert b.ids_hash(frame,indices)==canonical['context_ID_hash']
context=frame.iloc[indices][['application_id']+columns].copy()
context['target']=labels.target.iloc[indices].to_numpy(int)
if args.query:
    query=pd.read_parquet(args.query);query_identity={'query':b.sha(args.query)}
elif args.rows:
    positions=np.load(args.rows)[args.row_key]
    query=frame.iloc[positions].copy();query_identity={'rows_file':b.sha(args.rows),'rows_key':args.row_key}
else:
    assert len(frame)>len(labels),'This canonical feature table has no retained test rows'
    query=frame.iloc[len(labels):].copy();query_identity={'test_rows_from_canonical_feature_table':True}
assert 'target'not in query and query.application_id.is_unique
assert set(columns)<=set(query) and set(query.application_id).isdisjoint(context.application_id)
engine=Engine(checkpoint,columns,context)
raw=engine.fit_predict(np.arange(len(context)),query,canonical['seed'],cache_path=None)
assert engine.last_prediction_replay_error==0. and np.isfinite(raw).all();del engine
output={'application_id':query.application_id.to_numpy(str),'raw':raw}
identities={'model':b.sha(args.model),**query_identity}
if args.head:
    head=joblib.load(args.head);spline=joblib.load(args.spline)
    assert set(head['columns']+spline['columns'])<=set(query)
    corrected=b.corrected_probability(head,query,raw)[0]
    sp=b.predict_spline(spline,query)
    output.update(corrected=corrected,spline=sp,prediction=.75*corrected+.25*sp)
    identities.update(head=b.sha(args.head),spline=b.sha(args.spline))
else:output['prediction']=raw
assert np.isfinite(output['prediction']).all()
args.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.output,**output)
b.dump(dict(status='complete',query_rows=len(query),canonical_context_rows=len(context),seed=canonical['seed'],checkpoint_sha256=canonical['checkpoint_sha256'],identities=identities,output_sha256=b.sha(args.output),seconds=time.time()-started,source_sha256=b.sha(__file__),learned_head_or_spline_refitted=False),args.output.with_suffix('.complete.json'))
