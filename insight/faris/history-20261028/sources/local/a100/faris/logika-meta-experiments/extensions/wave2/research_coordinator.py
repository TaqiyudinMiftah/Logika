"""Second-wave search on the original clock; never reopens confirmation."""
from pathlib import Path
import argparse
from collections import Counter
import json
import os
import subprocess
import time
from scheduler_parallel import ParallelCoordinator, HOSTS
from metaexp.common import read,write,sha,code_hash,event


def progress(root,ledger,catalogue,final=False):
    root=Path(root);out=root/'reports';out.mkdir(exist_ok=True);records=ledger.all();accepted=[];effects=[];controls={}
    for record in records:
        if not record['verified']:continue
        score=read(root/'runs'/record['id']/'scores.json');row={'id':record['id'],**score}
        if record['mode']=='cv':
            accepted.append(row)
            if score['representation']==catalogue['control']:controls[(score['model'],score['device'])]=row
    accepted.sort(key=lambda r:r['pooled_oof']['ks'],reverse=True)
    for row in accepted:
        baseline=controls.get((row['model'],row['device']))
        if baseline is None:continue
        deltas=[x['ks']-y['ks'] for x,y in zip(row['per_fold'],baseline['per_fold'])]
        delta=row['pooled_oof']['ks']-baseline['pooled_oof']['ks']
        effects.append({'id':row['id'],'family':row['family'],'model':row['model'],'device':row['device'],
                        'control':baseline['id'],'ks_delta':delta,'fold_deltas':deltas,
                        'finding':'positive_cv' if delta>=.001 and sum(v>0 for v in deltas)>=2 else 'negative_cv' if delta<=-.001 and sum(v<0 for v in deltas)>=2 else 'inconclusive'})
    tested={s['recipe_id'] for s in accepted};new_ids={r['id'] for r in catalogue['recipes'] if r['representation'].get('wave')==2}
    summary={'wave':2,'final':final,'observed':time.time(),'state_counts':dict(Counter(r['state'] for r in records)),
             'new_pipeline_catalogue':len(new_ids),'new_verified_pipelines':len(tested&new_ids),
             'repeated_verified_controls':len(tested-new_ids),'deadline':read(root/'campaign.json')['deadline'],
             'research_only':True,'confirmation_access':False,'primary_nomination':'Unchanged in parent campaign',
             'families':{f:{'registered':sum(r['representation']['family']==f for r in catalogue['recipes'] if r['id'] in new_ids),
                            'verified':sum(s['family']==f and s['recipe_id'] in new_ids for s in accepted)} for f in catalogue['coverage']}}
    write(out/'summary.json',summary);write(out/'scoreboard_pipeline.json',accepted);write(out/'feature_effects.json',effects)
    write(out/'untested_recipes.json',[r['id'] for r in catalogue['recipes'] if r['id'] not in tested])
    write(out/'a100_handoff.json',{'scope':'Development-only second-wave recipes; these have no independent confirmation. Parent nomination remains frozen.',
                                 'recipes':[{'recipe':read(root/'runs'/r['id']/'task.json')['recipe'],'metrics':r['pooled_oof'],
                                             'device':r['device'],'preprocessing_audits':[f'runs/{r["id"]}/fold_{f}/preprocessing.json' for f in range(3)],
                                             'factory':'metaexp.pipeline.Preprocessor'} for r in accepted[:32]],
                                 'bank_id':read(root/'bank/manifest.json')['bank_id'],'source_hash':code_hash(root)})
    lines=['# Additional development-only experiments',f'Final: {final}. New verified pipelines: {len(tested&new_ids)} / {len(new_ids)}.',
           f'Attempt states: {summary["state_counts"]}. Original deadline: {summary["deadline"]}.',
           'Wave-one confirmation was already opened. Its nomination and evidence remain frozen. No new holdout validation is claimed.',
           'All ranked new pipelines have full aligned three-fold OOF, with training-fold transforms and independent saved-metric verification.',
           'Folds, retries and repeated controls are excluded from the new-pipeline count.',
           '## Highest development KS', '| ID | Model | Device | KS |','|---|---|---|---:|']
    lines.extend(f'| {r["id"]} | {r["model"]} | {r["device"]} | {r["pooled_oof"]["ks"]:.6f} |' for r in accepted[:20])
    (out/('final_report.md' if final else 'progress.md')).write_text('\n\n'.join(lines)+'\n')
    return summary


class ResearchCoordinator(ParallelCoordinator):
    def __init__(self,root):
        super().__init__(root);revision=read(self.root/'scheduler_revision.json')
        for name in ['research_coordinator.py','research_worker.py','design_extension.py']:
            if sha(self.root/name)!=revision['extension_sources'][name]:raise ValueError('Research extension source differs')
        if not self.cfg.get('research_only'):raise ValueError('Missing development-only boundary')

    def sync(self):
        super().sync()
        if time.time()-getattr(self,'last_extension_report',0)>600:
            progress(self.root,self.ledger,self.catalogue);self.last_extension_report=time.time()

    def run_search(self):
        end=self.phase_end(9.5);write(self.root/'phase.json',{'name':'development_only_second_wave','until':end})
        controls=[r for r in self.catalogue['recipes'] if r['representation']['id']==self.catalogue['control']]
        for recipe in controls:
            host='pc3' if recipe['model']=='cat' else 'pc2' if recipe['model']=='xgb' else 'pc1'
            self.submit(host,recipe,end)
            if recipe['model'] in ['cat','xgb','lgb']:
                self.submit('pc1' if recipe['model']!='lgb' else 'pc2',recipe,end,tid=recipe['id']+'_cpu',force_device='cpu')
        self.flush();self.sync();self.dispatch(end)
        while time.time()<end:
            self.sync();self.retry_delivery()
            if not any(r['state']=='assigned' for r in self.ledger.all()):break
            time.sleep(3)
        self.wait_phase(self.phase_end(9.55),mode=['cv'])

    def run(self):
        from metaexp.features import Bank
        from research_worker import install_guard
        install_guard();self.bank=Bank(self.root);self.hold_log=None
        try:
            # Rebuild the interrupted parent report inside this capped service.
            # Its confirmation predictions/statistics are already frozen; the
            # parent process uses its own unchanged source and has a separate report.
            parent=Path(self.cfg['parent_root'])
            if not (parent/'reports/summary.json').exists() or not read(parent/'reports/summary.json').get('final'):
                log=(self.root/'parent_reporting.log').open('ab')
                source='from pathlib import Path;from metaexp.reporting import report;report(Path.cwd(),final=True)'
                self.parent_report=subprocess.Popen([str(parent/'.venv/bin/python'),'-c',source],cwd=parent,stdout=log,stderr=log,
                                                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                log.close()
            for host in HOSTS:self.tx.call(host,{'action':'start','host':host})
            if time.time()<self.phase_end(9.5):self.run_search()
            if time.time()<self.phase_end(10.5):
                self.run_ensembles()
                # Sensitivity checks do not inflate the logical recipe count.
                candidates=self.accepted_pipelines()[:6]
                for i,(tid,score) in enumerate(candidates):
                    recipe=read(self.root/'runs'/tid/'task.json')['recipe'];host='pc2' if i%2==0 else 'pc3'
                    for mode,prefix in [('repeat','r_'),('temporal','t_')]:
                        self.submit(host,recipe,self.phase_end(11),mode=mode,tid=prefix+tid,parent_id=tid)
                self.wait_phase(self.phase_end(11),mode=['repeat','temporal'])
            write(self.root/'phase.json',{'name':'second_wave_verification','until':self.cfg['deadline']})
            self.sync()
            for host in HOSTS:self.tx.call(host,{'action':'stop'})
            time.sleep(3);self.sync();summary=progress(self.root,self.ledger,self.catalogue,True)
            # Ensemble scoreboards remain separate from fitted pipelines.
            for kind,prefix in [('fixed_ensemble','e_'),('nested_ensemble','n_')]:
                rows=[];pending=[];rejected=[]
                for p in (self.root/'runs').glob(prefix+'*/scores.json'):
                    if not (p.parent/'complete.json').exists():continue
                    if time.time()>self.cfg['deadline']-120:
                        pending.append(p.parent.name);continue
                    from metaexp.reporting import verify_run
                    try:score=verify_run(self.root,p.parent.name,self.bank)
                    except Exception as exc:
                        rejected.append({'id':p.parent.name,'error':str(exc)});continue
                    if score.get('kind')==kind:rows.append({'id':p.parent.name,**score})
                rows.sort(key=lambda r:r['pooled_oof']['ks'],reverse=True);write(self.root/'reports'/('scoreboard_'+kind+'.json'),rows)
                write(self.root/'reports'/(kind+'_verification.json'),{'verified':len(rows),'pending':pending,'rejected':rejected})
            write(self.root/'campaign_status.json',{'state':'completed','time':time.time(),'deadline':self.cfg['deadline'],'summary':summary,'errors':self.errors})
            try:self.tx.call('pc1',{'action':'cleanup_key'})
            except Exception as exc:event(self.root/'scheduler_parallel.jsonl',event='key_cleanup_pending',error=str(exc))
        finally:
            write(self.root/'research_guard.json',{'confirmation_access_permitted':False,'original_deadline':self.cfg['deadline']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();ResearchCoordinator(a.root).run()
