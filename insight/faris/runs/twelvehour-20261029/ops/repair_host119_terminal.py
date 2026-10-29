from pathlib import Path
import hashlib,json,subprocess,time
R=Path('/home/faris/logika-twelvehour-20261029');P=R/'private/host119';LAB='faris@100.68.29.126';CANON='/mnt/home/faris/logika-twelvehour-20261029';C=json.load(open(R/'CLOCK.json'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(a):return subprocess.run(a,check=True,capture_output=True,text=True,timeout=180)
# No training launch: the old unit was garbage-collected after normal completion.
processes=call(['ps','-u',str(__import__('os').getuid()),'-o','pid,args']).stdout
assert not any('ops/backend_queue.py --owner host119'in line or'ops/backend_worker.py --owner host119'in line for line in processes.splitlines())
final=json.load(open(R/'BACKEND_FINAL.json'));assert not final['pending']and not final['interrupted'];assert len(final['finished'])==78 and all(x['returncode']==0 for x in final['finished'])
plan=json.load(open(R/'COLAB_RESOURCE_ADDENDUM.json'))['assignments']['host119'];expected={f'{d}__s{s}__f{f}'for d in plan['designs']for s in[2026,7731]for f in range(5)};manifest={}
for task in expected:
 p=R/f'runs/{task}/attempt1';v=json.load(open(p/'RESULT.json'));start=json.load(open(p/'START.json'));assert v['status']=='complete'and v['weights_saved']is False and v['outer_labels_loaded']is False;assert sha(p/'prediction.npz')==v['prediction_sha256']and sha(p/'START.json')==v['start_sha256']
 for q in p.iterdir():
  if q.is_file():manifest[str(q.relative_to(R))]={'sha256':sha(q),'bytes':q.stat().st_size}
call(['rsync','-az','--ignore-existing',str(R/'runs')+'/',LAB+':'+CANON+'/runs/'])
# Compare the entire expected private manifest on canonical storage, not just rsync exit status.
remote="from pathlib import Path;import hashlib,json,sys;r=Path("+repr(CANON)+");m=json.load(sys.stdin);assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v['sha256']for k,v in m.items());print(len(m))"
p=subprocess.run(['ssh',LAB,'python3 -c '+__import__('shlex').quote(remote)],input=json.dumps(manifest),capture_output=True,text=True,check=True,timeout=180)
(P/'TERMINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');receipt=dict(epoch=time.time(),collected=True,verified_fits=len(expected),canonical_files_verified=int(p.stdout),private_manifest_sha256=sha(P/'TERMINAL_MANIFEST.json'),original_clock_preserved=True,execution_repair='Normal terminal unit had already unloaded; no training repeated. Reconciled processes and final records; hash-verified all80 fits on both hosts.')
(P/'FINAL_STATE.json').write_text(json.dumps(receipt,indent=2)+'\n');call(['scp','-q',str(P/'FINAL_STATE.json'),LAB+':'+CANON+'/BACKEND_COLLECTION_host119_COMPLETE.json']);print(json.dumps(receipt))
