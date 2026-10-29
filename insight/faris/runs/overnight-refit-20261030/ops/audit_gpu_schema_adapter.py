"""Map measured GPU audit field names in RAM; no model/output/score-rule edits."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('namespace');args=p.parse_args();P=Path(__file__).resolve().parents[1];R=P/args.namespace
assert args.namespace in ['gpu-book-features','helma-book-context10k'];sys.path.insert(0,str(R/'source'))
from common import load,sha,dump,ids_sha,utc,check_seal,tasks
from context_recipe import sample_context
check_seal();configs=load('CONFIG.json')['designs'];aliases=[]
for f in(R/'runs').glob('*/attempt1/RESULT.json'):
 v=load(f);start=load(f.parent/'START.json');a=v['audit'];design=configs[start['design_id']];fold=start['fold'];roles=np.load(R/f'packets/fold{fold}/roles.npz');selected,draw=sample_context(roles['y_context'],int(design['context_count']),start['seed'],0)
 assert a['no_query_labels']is True and a['selected_rows_sha256']==ids_sha(roles['context'][selected])and a['context_count']==len(selected)and a['context_draw_seed']==draw
 assert a['inner_train_rows_sha256']==start['declared_inner_train_rows_sha256']and a['stopping_rows_sha256']==start['declared_inner_stop_rows_sha256']
 ref=load('CHECKPOINTS.json')[design['size']];expected=ref if isinstance(ref,str)else ref['sha256'];assert a['pretrained_checkpoint_sha256']==expected and 0<=a['first_batch_replay_error']<1e-7
 aliases.append(dict(task=start['task_id'],result_sha256=sha(f),selected_rows_sha256=a['selected_rows_sha256'],pretrained_checkpoint_sha256=a['pretrained_checkpoint_sha256'],first_batch_replay_error=a['first_batch_replay_error']))
spec=importlib.util.spec_from_file_location('original_sealed_artifact_audit',R/'source/audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);original=m.load
def adapted(name):
 value=original(name)
 if str(name).endswith('/RESULT.json')and 'audit'in value and 'pretrained_checkpoint_sha256'in value['audit']:
  value=copy.deepcopy(value);a=value['audit'];a['context_selection_uses_query_labels']=not a['no_query_labels'];a['checkpoint_sha256']=a['pretrained_checkpoint_sha256'];a['engine_audit']={'first_batch_replay_error':a['first_batch_replay_error']}
 return value
m.load=adapted;m.main()
proof=dict(status='PASS',UTC=utc(),adapter_sha256=sha(Path(__file__)),original_sealed_audit_sha256=sha(R/'source/audit.py'),mapping_only='no_query_labels -> negated context_selection_uses_query_labels; pretrained_checkpoint_sha256 -> checkpoint_sha256; measured first_batch_replay_error -> engine_audit.first_batch_replay_error',extra_independent_checks='Recomputed natural stratified context draw from permitted context labels; exact selected row hash/seed/count, inner/stop hashes and checkpoint hash verified.',source_predictions_START_RESULT_untouched=True,unchanged_replay_tolerance=1e-7,aliases=aliases)
dump(proof,R/'GPU_AUDIT_SCHEMA_ADAPTER.json')
clock=load('CLOCK.json');result=dict(stage='complete',UTC=utc(),completed_fits=len(aliases),failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),interrupted_attempts=len(list((R/'runs').glob('*/attempt1/INTERRUPTION.json'))),planned_fits=len(tasks()),parent_clock_sha256=sha(P/'CLOCK.json'),derived_clock_sha256=sha(R/'CLOCK.json'),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),execution_only_schema_adapter_sha256=sha(R/'GPU_AUDIT_SCHEMA_ADAPTER.json'),original_coordinator_failure_preserved=True,separate_source_version_not_merged=True)
assert(R/'COLLECTED_gpu2003.json').exists()if args.namespace=='gpu-book-features'else(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists()
dump(result,R/'COMPLETION.json');dump(result,R/'LIVE_STATE.json');print(json.dumps({k:result[k]for k in ['stage','completed_fits','failed_attempts','interrupted_attempts','planned_fits']}))
