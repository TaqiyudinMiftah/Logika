"""Restore only explicit canonical logical aliases after filtered evidence sync.

Rsync include=*/ excludes symlink directories; these aliases reproduce the
canonical approved attempt2 mapping. They never turn failed output into success.
"""
from pathlib import Path
import hashlib,json,shutil

def apply_approved_aliases(root,require_predictions=True):
 root=Path(root);registry=root/'APPROVED_PRESTART_RETRIES.json'
 if not registry.exists():return[]
 restored=[]
 for task,approval in json.load(open(registry))['tasks'].items():
  assert approval['approved_attempt']==2
  parent=root/'runs'/task;approved=parent/'attempt2';alias=parent/'attempt1'
  if not(approved/'RESULT.json').exists():continue
  result=json.load(open(approved/'RESULT.json'));start=json.load(open(approved/'START.json'))
  assert result['status']=='complete'and result['attempt']==2 and start['attempt']==2 and result['task_id']==task and start['task_id']==task
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  assert sha(approved/'START.json')==result['start_sha256']
  if require_predictions:assert sha(approved/'prediction.npz')==result['prediction_sha256']
  if alias.is_symlink():assert alias.readlink()==Path('attempt2')
  else:
   if alias.exists():
    # Early incremental copies can contain the original pre-START failure.
    # Archive its exact approved hash before restoring the later canonical alias.
    assert alias.is_dir()and set(q.name for q in alias.iterdir())=={'FAILURE.json'},'Unexpected recorded attempt; review without overwriting'
    assert sha(alias/'FAILURE.json')==approval['original_failure_sha256']
    archive=root/'failures/filtered_prestart_retained'/task/'attempt1';archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():
     assert set(q.name for q in archive.iterdir())=={'FAILURE.json'}and sha(archive/'FAILURE.json')==approval['original_failure_sha256']
     shutil.rmtree(alias) # exact verified duplicate remains in archive
    else:alias.rename(archive)
   alias.symlink_to('attempt2',target_is_directory=True)
  restored.append(task)
 return restored
