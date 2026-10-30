"""Workstation-only sanitized final evidence publication; raw predictions private."""
from pathlib import Path
import importlib.util,json,hashlib,subprocess,time
R=Path(__file__).resolve().parent;repo=R.parents[3]/'.history-publish-20261028'
# Resolve from project cwd; never read a credential from a remote host.
repo=Path.cwd()/'.history-publish-20261028';dest=repo/'insight/faris/final-history-blend-20261030'
spec=importlib.util.spec_from_file_location('helper',repo/'tools/publish_experiment_history.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def run(args):return subprocess.run(args,cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
def g(*args):return run(['git','-c','gc.auto=0','-c','maintenance.auto=false',*args])
g('fetch','origin','main');assert g('rev-parse','HEAD')==g('rev-parse','origin/main'),'Inspect peer changes before publication'
exported=[];manifest=[]
for p in R.rglob('*'):
 rel=p.relative_to(R)
 if any(x in rel.parts for x in ['private','bank','__pycache__'])or p.suffix not in ['.json','.py','.md'] or not p.is_file():continue
 payload,policy=h.public_content(p)
 if payload is None:continue
 q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(payload);exported.append(str(q.relative_to(repo)));manifest.append(dict(path=str(rel),private_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),public_sha256=hashlib.sha256(payload).hexdigest(),policy=policy))
q=dest/'PUBLICATION_MANIFEST.json';q.write_text(json.dumps(dict(epoch=time.time(),files=manifest,raw_predictions_labels_credentials_and_fittedweights_not_published=True),indent=2)+'\n');exported.append(str(q.relative_to(repo)))
scanner=R.parent/'github-history-20261028/private/scanner/gitleaks';report=R/'private/FINAL_SECRET_SCAN.json'
subprocess.run([str(scanner),'dir',str(dest),'--config',str(repo/'.gitleaks.toml'),'--no-banner','--redact=100','--report-format','json','--report-path',str(report)],check=True);assert json.loads(report.read_text())==[]
for i in range(0,len(exported),200):g('add','-f','--',*exported[i:i+200])
g('commit','--only','-m','Record final historical blends, paired held candidates, negative results and audits','insight/faris/final-history-blend-20261030');g('fetch','origin','main');g('merge-base','--is-ancestor','origin/main','HEAD');g('push','origin','HEAD:main');head=g('rev-parse','HEAD');assert g('ls-remote','origin','refs/heads/main').split()[0]==head
receipt=dict(remote_verified_commit=head,metadata_files=len(exported),secret_findings=0,epoch=time.time());(R/'private/FINAL_PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
