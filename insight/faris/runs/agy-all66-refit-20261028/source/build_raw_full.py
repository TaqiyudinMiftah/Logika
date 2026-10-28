"""Extend the exact row-local historical feature recipe to unlabeled test records."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parents[1];A=Path('/mnt/home/faris/logika-faris-exploration')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=R/'features';out.mkdir(parents=True,exist_ok=True)
 source=R/'source/vendor/raw_domain_features_original.py';code=source.read_text()
 assert 'assert N == 70000'in code
 code=code.replace('DATA_DIR = "/home/parasite/Project/Competition/logika-2026/dataset"',f'DATA_DIR = {str(A/"exploration-12h-20261017/raw")!r}')
 code=code.replace('tr = pd.read_csv(f"{DATA_DIR}/train_applications.csv")','tr = pd.concat([pd.read_csv(f"{DATA_DIR}/train_applications.csv").drop(columns=["target"]),pd.read_csv(f"{DATA_DIR}/test_applications.csv")],ignore_index=True)')
 code=code.replace('assert N == 70000','assert N == 100000')
 start=code.index('out_path = ');code=code[:start]+'''\nout_path = str(OUTPUT_PATH)\nnp.savez_compressed(out_path,names=np.array(names),data=mat,application_id=apps["application_id"].to_numpy(str))\n'''
 ns={'__name__':'exact_raw_full','OUTPUT_PATH':out/'raw_domain_features.npz'};exec(compile(code,str(source),'exec'),ns)
 reference=np.load('/mnt/home/faris/logika-joint-20261028/raw_domain_features.npz',allow_pickle=True);result=np.load(out/'raw_domain_features.npz')
 assert np.array_equal(result['names'],reference['names'])
 assert np.array_equal(result['data'][:70000],reference['data'],equal_nan=True),'Row-local training features changed while adding test rows'
 assert np.array_equal(np.sum(result['data'][:,[list(result['names']).index(c)for c in ['pop_A','pop_B','pop_C1','pop_C2']]],axis=1),np.ones(100000))
 report={'status':'passed','rows':100000,'columns':len(result['names']),'training70000_exact':True,'target_labels_opened':False,'source_original_sha256':sha(source),'source_wrapper_sha256':sha(__file__),'result_sha256':sha(out/'raw_domain_features.npz'),'changes':'Extend applications with30k test; drop training target before transforms; unchanged row-local math and original train row order.'}
 (out/'RAW_PARITY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
