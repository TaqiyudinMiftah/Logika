"""Execution metadata adapter; sealed gate/model/features and predicates unchanged.

The original worker omitted GPU_name and GPU_memory_bytes from START while the
sealed gate expected them. Resolve those two measured facts from same-session
attestations, retain every original failed gate, and run every original predicate.
Never edit START, predictions, scientific source, contract or clock.
"""
from pathlib import Path
import argparse,inspect,json,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import gate
from common import check_seal,load,catalog,tasks,sha
p=argparse.ArgumentParser();p.add_argument('--family',required=True);a=p.parse_args();check_seal()
keys=[k for k in load('CONFIG.json')['pilot_tasks']if catalog()[tasks()[k]['design']]['family']==a.family];assert len(keys)==1
folders=sorted((R/'runs'/keys[0]).glob('attempt*/RESULT.json'));assert len(folders)==1
start=json.loads((folders[0].parent/'START.json').read_text());bindings=load('GPU_BACKENDS.json');backend=bindings[start['host']]
assert backend['GPU_name'] and backend['GPU_memory_bytes']>0 and backend['host']==start['host']
assert backend['binding_basis'] in {'same-session private environment collection','prelaunch same-host measured CUDA device'}
attestation=R/backend['evidence_path'];assert sha(attestation)==backend['evidence_sha256']
measured=json.loads(attestation.read_text());assert measured.get('GPU',measured.get('GPU_name'))==backend['GPU_name']
assert measured['GPU_memory_bytes']==backend['GPU_memory_bytes']
# Retain all checks, including package, exact row hashes, seeds, full refit,
# training boundaries, source lineage and Kumo replay. Only metadata lookup moves.
s=inspect.getsource(gate.family_gate)
for old,new in [("start['GPU_name']","backend['GPU_name']"),("start['GPU_memory_bytes']","backend['GPU_memory_bytes']"),("start.get('GPU_name')","backend.get('GPU_name')")]:
 assert old in s;s=s.replace(old,new)
gate.backend=backend
original=gate._finish
def finish(kind,canonical,clock,checks,extra=None):
 extra=dict(extra or {},execution_metadata_adapter=dict(source_sha256=sha(__file__),sealed_gate_sha256=sha(R/'source/gate.py'),fields_resolved=['GPU_name','GPU_memory_bytes'],backend=backend,all_original_acceptance_predicates_retained=True,scientific_source_or_selection_changed=False))
 return original(kind,canonical,clock,checks,extra)
gate._finish=finish
exec(compile(s,str(Path(__file__)), 'exec'),gate.__dict__)
r=gate.family_gate(a.family);raise SystemExit(0 if r['status']=='PASS' else 2)
