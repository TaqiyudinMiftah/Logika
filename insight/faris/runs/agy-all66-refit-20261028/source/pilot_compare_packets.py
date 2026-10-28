"""Audit declared cross-backend replay tolerances after predictions return locally."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def ks(y,p):
    z=np.argsort(p,kind='mergesort');ys=y[z];ps=p[z];d=np.abs(np.cumsum(ys==1)/np.sum(ys==1)-np.cumsum(ys==0)/np.sum(ys==0));return float(np.max(d[np.r_[ps[:-1]!=ps[1:],True]]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packets',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--target-npz',help='Original globally exposed70k label file. Keep private; not needed onColab.');ap.add_argument('--target-key',default='target');ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.packets);predroot=Path(a.predictions)
    manifest=json.loads((root/'MANIFEST.json').read_text());labels=np.load(a.target_npz)[a.target_key]if a.target_npz else None;checks=[]
    for p in manifest['packets']:
        rel=Path(p['packet']);packet=root/rel;output=predroot/rel
        for name,e in p['files'].items():assert sha(packet/name)==e['sha256'],(rel,name,'changedpacket')
        meta=json.loads((packet/'meta.json').read_text());ref=np.load(packet/'reference.npz');new=np.load(output/'prediction.npz');rows=np.load(packet/'query_rows.npy');assert np.array_equal(rows,ref['rows'])and np.array_equal(rows,new['rows'])
        y=new['prediction'];r=ref['prediction'];assert y.shape==r.shape and np.isfinite(y).all()and(y>=0).all()and(y<=1).all()
        tolerance=meta['cross_backend_tolerance'];maxerr=float(np.max(np.abs(y-r)));meanerr=float(np.mean(np.abs(y-r)));result={'packet':str(rel),'query_rows':len(rows),'exact_rows':True,'packet_hashes_verified':True,'max_abs_prediction_error':maxerr,'mean_abs_prediction_error':meanerr,'tolerances_predeclared_in_packet':tolerance,'prediction_tolerances_passed':maxerr<=tolerance['max_abs_prediction_error']and meanerr<=tolerance['mean_abs_prediction_error'],'reference_prediction_sha256':sha(packet/'reference.npz'),'new_prediction_sha256':sha(output/'prediction.npz'),'bitwise_reproduction_claimed':False}
        if labels is not None:
            result.update(reference_KS=ks(labels[rows],r),new_KS=ks(labels[rows],y));result['abs_KS_error']=abs(result['reference_KS']-result['new_KS']);result['passed']=result['prediction_tolerances_passed']and result['abs_KS_error']<=tolerance['abs_KS_error']
        else:result['passed']=None;result['qualification']='Metric check remains pending until locally available labels are provided; no query labels were uploaded.'
        checks.append(result)
    receipt={'status':'PASSED'if all(r['passed']is True for r in checks)else'FAILED_OR_PENDING','checks':checks,'scope':'Portability replay of exposed fold0 predictions; no independent score-gain evidence.'};Path(a.out).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
    if labels is not None:assert receipt['status']=='PASSED','Cross-backend gate failed'
if __name__=='__main__':main()
