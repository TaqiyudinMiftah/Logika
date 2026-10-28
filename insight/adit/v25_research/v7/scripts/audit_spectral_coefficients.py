"""Read saved V167 heads to inspect duplicated spectral missing indicators."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'


def main():
    rows=[]
    for fold in range(5):
        p=V/'runs/V167'/f'fold{fold}_correction.joblib'
        bundle=joblib.load(p)
        head=bundle['head']
        nums=list(head['encoder'].transformers_[0][2])
        width=head['encoder'].named_transformers_['smooth'].named_steps['basis'].n_features_out_//len(nums)
        ix=[2+len(nums)*width+i for i,c in enumerate(nums)
            if c.startswith('beh_spectral_') and c!='beh_spectral_window_supported']
        beta=np.asarray(head['coefficients'])[ix]
        assert len(ix)==10
        rows.append(dict(fold=fold,missing_indicators=len(ix),coefficient_sum=float(beta.sum()),
                         coefficient_min=float(beta.min()),coefficient_max=float(beta.max()),
                         std=float(beta.std()),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    out=V/'audit_noise/reports'
    out.mkdir(parents=True,exist_ok=True)
    (out/'SPECTRAL_DUPLICATE_COEFFICIENT_AUDIT.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
