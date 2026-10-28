"""Replay the 4096 previously registered fixed blends, without model fitting."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import verify_predictions as audit
ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT.parent/'exploration-12h-20261017'
def main():
    start=time.time();audit.init();sys.path.insert(0,str(CAMPAIGN/'source'))
    import utils
    source=CAMPAIGN/'source/utils.py';board_path=CAMPAIGN/'reports/scoreboard_fixed.json'
    map_path=CAMPAIGN/'ensemble_sources.json';board=json.loads(board_path.read_text());mapping=json.loads(map_path.read_text())
    arrays={};sources={}
    for key,value in mapping.items():
        path=CAMPAIGN/'evidence/champion_oof.npz' if key=='champion' else Path(value['folder'])/'validation.npz'
        with np.load(path,allow_pickle=False) as z:
            arrays[key]=z['existing_feature_ensemble' if key=='champion' else 'prediction']
            assert arrays[key].shape==audit.Y.shape
            if key!='champion':assert np.array_equal(z['rows'],audit.ROWS) and np.array_equal(z['fold'],audit.FOLDS)
        sources[key]=dict(path=str(path),sha256=audit.sha(path))
    audit.write(ROOT/'validation/fixed_replay_plan.json',dict(total=len(board),training=False,
        source=str(source),source_sha256=audit.sha(source),board_sha256=audit.sha(board_path),
        mapping_sha256=audit.sha(map_path),parents=sources,
        scope='Recalculation of existing fixed ensembles, not additional trained experiments.'))
    output=ROOT/'validation/fixed_replay.jsonl';assert not output.exists(),'Preserve previous evidence'
    counts={}
    with output.open('w',buffering=1) as file:
        for i,item in enumerate(board):
            result=dict(id=item['id'],configuration=item['spec'])
            try:
                p=utils.combine(item['spec'],arrays);pooled=audit.metric(audit.Y,p)
                assert abs(pooled['KS']-item['ks'])<1e-10,'KS mismatch'
                folds=[audit.metric(audit.Y[audit.FOLDS==f],p[audit.FOLDS==f]) for f in range(3)]
                saved=CAMPAIGN/'reports'/(item['id']+'.npy')
                if saved.exists():assert np.max(abs(np.load(saved)-p))<1e-14,'Saved shortlist mismatch'
                result.update(state='verified',pooled=pooled,folds=folds,
                    prediction_array_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
                    score_verification_only=True,validation_integrity_not_implied=True)
            except Exception as e:result.update(state='invalid',error=repr(e))
            file.write(json.dumps(result,allow_nan=False)+'\n');counts[result['state']]=counts.get(result['state'],0)+1
            if (i+1)%100==0:audit.write(ROOT/'validation/fixed_replay_status.json',dict(status='running',done=i+1,total=len(board),states=counts,seconds=time.time()-start))
    audit.write(ROOT/'validation/fixed_replay_status.json',dict(status='complete',done=len(board),total=len(board),states=counts,seconds=time.time()-start,training=False))
if __name__=='__main__':main()
