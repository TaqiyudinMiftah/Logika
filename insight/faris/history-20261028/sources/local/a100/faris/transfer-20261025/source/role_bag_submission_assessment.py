"""Apply the prespecified last-slot rule; no fitting or new resampling."""
import numpy as np
import baseline as b
from robustness import R,read
r=read(R/'ROLE_BAG_RESULT.json');boot=np.load(R/'ROLE_BAG_BOOTSTRAP.npz')
point=r['metrics']['fivebag_direct_KS']['KS']-r['metrics']['fivebag_ordinary']['KS']
draws=boot['direct_KS__matched_onebag']-boot['ordinary__matched_onebag']
interval=np.quantile(draws,[.005,.995]).tolist()
all_intervals={k:np.quantile(boot[k],[.005,.995]).tolist()for k in boot.files}
all_intervals['direct_KS_vs_ordinary_fivebag']=interval
required=['direct_KS__matched_onebag','direct_KS__previous_B','direct_KS_vs_ordinary_fivebag']
eligible=r['selected']=='direct_KS'and point>0 and all(all_intervals[k][0]>0 for k in required)
b.dump(dict(status='complete',eligible_for_full_refit_and_last_slot=bool(eligible),eligible_for_research_full_refit=(r['selected'] is not None),eligible_for_last_slot=bool(eligible),native_H010_selected=r['selected'],directKS_vs_ordinary_fivebag_KS_delta=point,five_contrast_adjusted95percent_interval=interval,all_five_contrast_intervals=all_intervals,rule_sha256=b.sha(R/'H010_SUBMISSION_RULE.json'),selection_evidence_sha256=b.sha(R/'ROLE_BAG_RESULT.json'),independent_confirmation=False,source_sha256=b.sha(__file__)),R/'ROLE_BAG_SUBMISSION_ASSESSMENT.json')
