"""Effective user feature policy; historical result files remain unchanged."""
import copy,json
from pathlib import Path
V=Path(__file__).resolve().parents[1]

def policy():
 p=V/'FEATURE_POLICY.json'
 return json.loads(p.read_text()) if p.exists() else {}

def effective_result(value):
 r=copy.deepcopy(value);spec=r['spec'];fields=spec.get('snapshot_fields') or (['max_days_past_due'] if spec.get('quarantined') or 'dpd_research' in spec.get('families',[]) else [])
 has_dpd=any('past_due' in x for x in fields)
 if policy().get('allow_DPD_without_quarantine') and has_dpd and 'amount_paid' not in fields:
  r['original_evidence_at_completion']=r['evidence']
  r['DPD_policy']='Allowed by explicit user instruction; snapshot uncertainty documented, not certified safe'
  if r['evidence']=='QUARANTINED_SNAPSHOT_PROVENANCE_UNVERIFIED':
   promising=r['confidence_interval']['delta_KS'][0]>0 and r['fold_wins']>=4 and r['delta_AUC']>=-.0005
   r['evidence']='PROMISING_REQUIRES_INDEPENDENT_VALIDATION' if promising else 'INCONCLUSIVE_OR_NEGATIVE'
  r['effective_quarantined']=False;spec['label']=spec['label'].replace('QUARANTINED ','')
 return r
