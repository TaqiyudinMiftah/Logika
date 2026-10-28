"""Execute the inherited label-free cohort builder with all70k as context.

This is a new owned build. Historical source, clocks and seals are untouched.
Only membership, destination and this authorized execution gate differ.
"""
from pathlib import Path
import ast,hashlib,json,sys,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];A=Path('/mnt/home/faris/logika-faris-exploration');F=A/'submission-full-periodicity-20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=R/'prior70k';root.mkdir(parents=True,exist_ok=True)
 assert not(root/'fold0').exists(),'Preserve previous attempts; reconcile before retrying'
 src=F/'feature_build/source/build.py';source=src.read_text();source=source.replace('len(context)==64000','len(context)==70000').replace("sum())==64000","sum())==70000").replace('context_rows=64000','context_rows=70000')
 # Entry-point and ROOT assignment are explicitly excluded; no old campaign
 # execution gate or output path is reused.
 tree=ast.parse(source);nodes=[]
 for node in tree.body:
  if isinstance(node,ast.If)and'__name__'in ast.unparse(node.test):continue
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ROOT'for t in node.targets):continue
  if isinstance(node,ast.FunctionDef)and node.name=='check':continue
  nodes.append(node)
 root.joinpath('vendor').mkdir(exist_ok=True)
 for p in(F/'feature_build/vendor').glob('*.py'):shutil.copy2(p,root/'vendor'/p.name)
 input_root=root/'input';(input_root/'data').mkdir(parents=True,exist_ok=True);(input_root/'artifacts/fold0').mkdir(parents=True,exist_ok=True)
 for name,target in [('data/train_features.parquet',A/'ranking-pipeline-20261020/input/data/train_features.parquet'),('BASELINE_LOCK.json',A/'ranking-pipeline-20261020/input/BASELINE_LOCK.json')]:
  path=input_root/name
  if not path.exists():path.symlink_to(target)
 np.savez_compressed(input_root/'artifacts/fold0/indices.npz',context=np.arange(70000),validation=np.array([],dtype=np.int64))
 plan=json.loads((F/'feature_build/plan.json').read_text());plan.update(input=str(input_root),full_data_refit=True,preprocessing_only=True,model_training_admitted=False,historical_metric_recalculation_complete=True)
 (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 provenance={'source_original':str(src),'source_original_sha256':sha(src),'wrapper_sha256':sha(__file__),'training_membership':'All70000labeledtrainingapplicants; labels replaced withconstant0 before feature estimation','validation_rows':0,'test_rows':30000,'shared_across_folds_reason':'Identical all70k context membership, inheritedbank, rawsources and deterministic priorfunctions; fold modelseedsremainseparate','old_campaign_modified':False}
 (root/'BUILD_PLAN.json').write_text(json.dumps(provenance,indent=2)+'\n')
 def gate(plan):
  assert (R/'CONFIG.json').exists()and(R/'PROTOCOL.json').exists(),'Loggednewrefitneeded'
  assert not(R/'STOP').exists(),'Newrefitstopped'
 ns={'__name__':'new_prior70k','__file__':str(src),'ROOT':root,'check':gate};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns);ns['build'](0)
 (root/'COMPLETE.json').write_text(json.dumps({**provenance,'features_sha256':sha(root/'fold0/features.parquet'),'status':'complete'},indent=2)+'\n')
if __name__=='__main__':main()
