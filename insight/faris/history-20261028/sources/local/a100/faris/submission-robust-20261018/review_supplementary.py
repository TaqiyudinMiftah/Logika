"""Reconcile the older depth runs and historical unaligned evidence separately."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import review_all as review
from utils import read,write,sha


def main():
    review.init();root=review.ROOT;depth=Path('/home/faris/faris/depth-20261017');audits=[];records=[]
    for name in ['campaign','campaign-parallel']:
        for file in sorted((depth/name/'records').glob('*.json')):
            row=read(file);records.append({'source':str(file),'sha256':sha(file),'job':row})
            if row['status']!='complete' or row['protocol']!='screen':continue
            folder=depth/name/row['directory'];out=read(folder/'outcome.json')
            item={'id':'depth/'+name+'/'+row['job_id'],'original_id':row['job_id'],'origin':'depth','model':row['family'].split('_')[0],
                'family':row['family'],'reported_ks':out['evaluation']['score'],'folder':str(folder),'prediction_path':str(folder/'validation.npz')}
            audits.append(review.assess(item))
    original=read(root/'evidence/all_prediction_audits.json');original=[r for r in original if r['origin']!='depth'];original+=audits
    write(root/'evidence/all_prediction_audits.json',original)
    additions=[]
    for r in original:
        if r['audit_state']!='verified' or r['degenerate_fold'] or r['quality_warnings']:continue
        for s in r['additions']:
            additions.append({'id':r['id'],'model':r['model'],'family':r['family'],**s,'mean_fold_delta':float(np.mean(s['fold_delta'])),'worst_fold_delta':min(s['fold_delta'])})
    additions.sort(key=lambda s:(min(s['fold_delta']),s['delta']),reverse=True)
    write(root/'evidence/descriptive_additions.json',additions)
    summary=read(root/'review_summary.json');summary.update(audited=len(original),depth_records=len(records),depth_verified=sum(r['audit_state']=='verified' for r in audits),top_worst_fold=additions[:30])
    summary['states']={state:sum(r['audit_state']==state for r in original) for state in sorted({r['audit_state'] for r in original})}
    write(root/'review_summary.json',summary)
    history=Path('/home/faris/faris/logika-meta-experiments/meta-20261016-v1/history_inventory.json')
    write(root/'evidence/supplementary_inventory.json',{'depth_records':records,'history_inventory_source':str(history),'history_inventory_sha256':sha(history),
        'scope':'Older Logika and original Kumo results retain their original protocols; not silently aligned or treated as fresh validation.',
        'history_inventory':read(history)})
    print(json.dumps({'audited':len(original),'depth_records':len(records),'depth_verified':summary['depth_verified'],'top':additions[:8]},indent=2))


if __name__=='__main__':main()
