"""Scientific figures from the saved post-campaign audit; no model execution."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
labels={'champion':'Existing champion','pc2_job_000623':'EBM','pc2_job_000853':'Logistic stack','pc2_job_000855':'Ridge stack','pc3_job_000424':'Account elastic net','pc3_job_000846':'Bureau TabM','new_logit':'New ensemble'}
corr=pd.read_csv(root/'prediction_spearman.csv',index_col=0).loc[list(labels),list(labels)]
fig,ax=plt.subplots(figsize=(9.5,8));im=ax.imshow(corr,vmin=.95,vmax=1,cmap='YlGnBu');ax.set_xticks(range(len(labels)),labels.values(),rotation=35,ha='right');ax.set_yticks(range(len(labels)),labels.values())
for i in range(len(labels)):
 for j in range(len(labels)):ax.text(j,i,f'{corr.iloc[i,j]:.4f}',ha='center',va='center',fontsize=10,color='white' if corr.iloc[i,j]>.982 else '#152733')
ax.set_title('Different learners, highly similar applicant rankings',pad=20,fontweight='bold');fig.colorbar(im,ax=ax,fraction=.04,pad=.04,label='Spearman rank correlation');fig.text(.5,.02,'56,000 development OOF predictions. Correlation describes overlap; it does not measure causal feature value.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(root/'prediction_overlap.png',dpi=180);fig.savefig(root/'prediction_overlap.svg');plt.close(fig)

v=json.loads((root/'fusion_and_ablation.json').read_text()); names=['pc2_job_000623','pc2_job_000853','pc2_job_000855','pc3_job_000424','pc3_job_000846']; protocols=['screen','reseed','repeat'];colors=['#264e70','#e07a3f','#4f8c70']
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,6));xx=np.arange(len(names))
for j,p in enumerate(protocols):
 values=[next(x['delta_vs_full_new'] for x in v if x['protocol']==p and x['diagnostic']=='without_'+n) for n in names];ax.bar(xx+(j-1)*.24,values,width=.23,label=['Original CV','New seed','New partition'][j],color=colors[j])
ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(xx,[labels[n] for n in names],rotation=25,ha='right');ax.set_ylabel('KS change after removing stream');ax.set_title('TabM contributes consistently; two stacks overlap',fontweight='bold');ax.legend(fontsize=9)
v=json.loads((root/'cohort_stability.json').read_text());names=['no_repayment','has_repayment','no_account','has_account'];xx=np.arange(len(names))
for j,p in enumerate(protocols):
 values=[next(x['delta'] for x in v if x['protocol']==p and x['cohort']==n) for n in names];bx.bar(xx+(j-1)*.24,values,width=.23,color=colors[j],label=['Original CV','New seed','New partition'][j])
bx.axhline(0,color='#333333',lw=.8);bx.set_xticks(xx,['No repayment','Has repayment','No account','Has account'],rotation=25,ha='right');bx.set_ylabel('New ensemble KS minus champion KS');bx.set_title('Apparent specialist advantages change on repeat',fontweight='bold');fig.text(.5,.015,'Post-hoc diagnostics from existing development predictions. No confirmation evaluation or model fitting.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.06,1,1));fig.savefig(root/'validation_tradeoffs.png',dpi=180);fig.savefig(root/'validation_tradeoffs.svg');plt.close(fig)
print('Saved two figures as PNG and SVG.')
