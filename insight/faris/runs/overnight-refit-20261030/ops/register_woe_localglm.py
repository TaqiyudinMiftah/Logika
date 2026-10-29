from pathlib import Path
import json,hashlib,shutil,py_compile
P=Path('a100/faris/overnight-refit-20261030');F=P/'woe-feature-cann';R=P/'woe-localglm-cann';R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(F/n,R/n,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','AGENTS.md','PARENT_CLOCK_REFERENCE.json']:(R/n).write_bytes((F/n).read_bytes())
s=(F/'source/cpu_models.py').read_text();assert 'self.head=torch.nn.Linear(width,1)'in s
s=s.replace('self.head=torch.nn.Linear(width,1)','self.head=torch.nn.Linear(width,features)\n        self.intercept=torch.nn.Parameter(torch.zeros(()))').replace('def forward(self,x):return self.head(self.body(x)).squeeze(-1)','def forward(self,x):return self.intercept+(self.head(self.body(x))*x).sum(dim=-1)')
s=s.replace("model_family='Crossfit_WoE_Bernoulli_CANN'","model_family='Crossfit_WoE_LocalGLMnet'").replace("formula='sigmoid(frozenGLMlogit+zero_initialized_MLP_residual)'","formula='sigmoid(frozenGLMlogit+beta0+sum_j(beta_j(x)*x_j))'")
(R/'source/cpu_models.py').write_text(s)
c=json.loads((F/'CONFIG.json').read_text());designs={}
for old,v in c['designs'].items():
 name=old.replace('woe_','woe_localglm_',1);d=dict(v);d.update(admission_group=name,family='Crossfit_WoE_LocalGLMnet',hypothesis='Test the interaction between own-label-excluded smoothed WoE features and input-dependent regression attention. Exactly paired with64WOE scalar-headCANN controls; earlier rawLocalGLM highest current localpoint is suggestive development evidence, not independent support.',sources=['user:978-3-031-12409-9.pdf#PDFpages504-506/Definition11.12','user:Credit-Scoring--Theory,-Methods,-and-Practice.pdf#PDFpage197']);designs[name]=d
c.update(campaign='woe-localglm-cann',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs],priority_designs=list(designs),expected_fits=len(designs)*10,notes='64head-structure interactions/640plannedfits. All other recipes identical to existingWOECANN controls. Original clock and same perdesign runtime gates. No noveltyclaim forWOE orLocalGLM individually.')
for k,v in list(c.items()):
 if isinstance(v,list)and any(isinstance(x,str)and x.startswith('woe_')for x in v):c[k]=list(designs)
(R/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
plan=json.loads((F/'EXECUTION_PLAN.json').read_text())
for v in plan['owners'].values():
 v['designs']=[x.replace('woe_','woe_localglm_',1)for x in v['designs']];v['pilot_tasks']=[x.replace('woe_','woe_localglm_',1)for x in v['pilot_tasks']]
(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
contract=json.loads((F/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v10-woe-localglm-paired64',CPU_protocol_change='Replace scalarMLPresidual with beta0+sum_j(beta_j(x)*x_j), zero initialization; same cross-fitted risk encoder, network body, loss, penalty, stopping and freshrefit. Exact64earlier controls reused.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
s=(F/'ops/woe_coordinator.py').read_text().replace('woe0630','woelocalglm0630').replace("P/'localglm-cann/private/owner_metadata'","P/'woe-feature-cann/private/owner_metadata'").replace('Crossfit_WoE_Bernoulli_CANN','Crossfit_WoE_LocalGLMnet');(R/'ops/woe_localglm_coordinator.py').write_text(s)
(R/'README.md').write_text('''# Cross-fitted credit-risk features with LocalGLMnet
64configurations/640plannedfits, not completion. Exactly paired with woE-feature-cann (directory woe-feature-cann) scalarheadcontrols; only structural residualchanges to beta0+dot(beta(x),x). This tests an interaction of two already sourced ideas, not a new algorithm or independent replication. Twofixed99views, GLM/intercept offsets, widths64/128,8/16bins,24/100priorstrength, append/encoded-only; lossBCE, residualenergy.1, lr.0003, architecturebody/dropout/optimizer/stopping unchanged. Wuthrich/Merz PDF504–506 and suppliedCreditScoring197. Same label-independent3fold inner crossfit42500 excluding ownlabel;7500stopper/6kmeta/14kouter labels excluded. Fresh50kmodel retains encoder; all70k cases adaptively reused development. Earlier rawLocalGLM.5316414268 is a point estimate withconditionalCIcrossingzero; this is an interactionhypothesis, not evidence of supported gain. Perowner waits previousWOEownerFINAL then4/4/6/6CPUslots, originaladmission22:30/model23:00/allcompute23:30UTC ceiling. Gatesfinite/boundary/strict5e-6batchreplay and conservative1.5runtime/300sec margin; no retries or shortened epochs. No newpaidallocation or foreignwork interference. Source/config/protocol published before fits; no fittedweights/KVcaches. Higher audited current candidate requires fresh70krefit and exact meta/CSVreplay before scheduled06:25WIB upload; currentREADYnever downgraded.
''')
for p in (R/'source').glob('*.py'):py_compile.compile(str(p),doraise=True)
(R/'SOURCE_SEAL.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((R/'source').rglob('*.py'))},indent=2)+'\n')
print('Prepared',len(designs),'paired structural interactions')
