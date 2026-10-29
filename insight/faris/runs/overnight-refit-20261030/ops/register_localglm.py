from pathlib import Path
import itertools,json,hashlib,shutil,py_compile
P=Path('a100/faris/overnight-refit-20261030');F=P/'book-feature-cann';R=P/'localglm-cann';R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(F/n,R/n,dirs_exist_ok=True)
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','EXECUTION_PLAN.json','AGENTS.md','PARENT_CLOCK_REFERENCE.json']:(R/n).write_bytes((F/n).read_bytes())
s=(F/'source/cpu_models.py').read_text().replace('from book_features import BookInput as FrozenInput','from book_features import BookInput\nfrom cpu_offset import FrozenInput')
s=s.replace("encoder=FrozenInput(mode=spec['book_transform']).fit(frame,ctx[inner])", "encoder=(FrozenInput() if spec['book_transform']=='raw' else BookInput(mode=spec['book_transform'])).fit(frame,ctx[inner])")
s=s.replace('self.head=torch.nn.Linear(width,1)','self.head=torch.nn.Linear(width,features)\n        self.intercept=torch.nn.Parameter(torch.zeros(()))')
s=s.replace('def forward(self,x):return self.head(self.body(x)).squeeze(-1)','def forward(self,x):return self.intercept+(self.head(self.body(x))*x).sum(dim=-1)')
s=s.replace("model_family='BookFeature_Bernoulli_CANN'","model_family='Bernoulli_LocalGLMnet_offset',structural_formula='offset+beta0+sum_j(beta_j(x)*x_j)'")
(R/'source/cpu_models.py').write_text(s)
c=json.loads((F/'CONFIG.json').read_text());designs={}
for view,offset,width,residual,transform in itertools.product(['actuarial99','global99'],['intercept','glm'],[64,128],[0.,.1,.5],['raw','asinh','both']):
 name=f'localglm_{view}_{offset}_w{width}_r{str(residual).replace(".","p")}_{transform}'
 designs[name]=dict(admission_group=name,view=view,offset_mode=offset,width=width,glm_C=.1,residual_penalty=residual,learning_rate=.0003,book_transform=transform,max_seconds=900,parameters={},family='Bernoulli_LocalGLMnet',resource='cpu',hypothesis='Feature-dependent regression attention adds explicit multiplicative interactions to CANN. Compare raw and robust numerical maps with exact earlier scalar-head CANN controls. No gain assumed.',sources=['user:978-3-031-12409-9.pdf#PDFpages504-506/Definition11.12'])
c.update(campaign='localglm-cann',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs],priority_designs=list(designs),expected_fits=len(designs)*10,notes='72prospective LocalGLMnet structural adaptations,720plannedfits; starts perowner after bookfeatureCANN; retains same overnightclock.')
for k,v in list(c.items()):
 if isinstance(v,list)and any(isinstance(x,str)and x.startswith('book_')for x in v):c[k]=list(designs)
(R/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n')
plan=json.loads((F/'EXECUTION_PLAN.json').read_text());owners=list(plan['owners']);cycle=sum(([o]*plan['owners'][o]['slots']for o in owners),[])
for owner in owners:
 names=[d for j,d in enumerate(designs)if cycle[j%len(cycle)]==owner];plan['owners'][owner].update(designs=names,pilot_tasks=[f'{d}__s2026__f0'for d in names],planned_fits=len(names)*10)
(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
contract=json.loads((F/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v7-localglm-fixed72',CPU_protocol_change='Bernoulli offset+beta0+dot(beta(x),x) with q-dimensional attention output; same scalar CANN architecture controls, loss and stopping.',feature_policy='Raw or previously predeclared bookfeatures. No other featureselection. Maps42500inner only. The model attention is not uniquely identifiable, so no causal explanation claim.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
s=(F/'ops/book_coordinator.py').read_text().replace('book0630','localglm0630').replace("P/'neural-focal-cann/private/owner_metadata'","P/'book-feature-cann/private/owner_metadata'").replace('BookFeature_CANN','Bernoulli_LocalGLMnet')
(R/'ops/localglm_coordinator.py').write_text(s)
(R/'README.md').write_text('''# Book-derived LocalGLMnet
Wuthrich/Merz PDFpages504–506, Definition11.12 and Eq11.37 define input-dependent regression attention, beta0+sum_j beta_j(x)*x_j. This experiment adapts that structure to Bernoulli logits with either a frozen train-onlyGLM or intercept offset, zero-initialized coefficient head and trainablezero intercept. TwoGELUhiddenlayers/dropout.1, widths64/128, AdamWlr.0003/weightdecay.01, residual-energypenalty0/.1/.5. This is a structural LocalGLMnet adaptation, not a claim to reproduce all book training choices. No causal explanation/identifiability claim. Twoexisting99-columnviews with raw,asinh orbothmaps give72configurations/720plannedfits. Earlier scalar-headCANN controls reused. Source/gradient/zerohead/replay checks before admission. Perowner waits previousbook-feature queue then fills4/4/6/6slots, same absolute06:30WIB deadline, no oldclock changes. Same42500inner/7500stopper/50krefit/6kmeta/14kouter validation; all70k adaptive development. Bestcompleteauditedhigherlocal candidate is fresh70k-refitted and may replace READY before deadline; no V25fallback.
''')
for p in list((R/'source').glob('*.py'))+list((R/'ops').glob('*.py')):py_compile.compile(str(p),doraise=True)
(R/'SOURCE_SEAL.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((R/'source').rglob('*.py'))},indent=2)+'\n')
print('Prepared',len(designs),'LocalGLMnet configurations')
