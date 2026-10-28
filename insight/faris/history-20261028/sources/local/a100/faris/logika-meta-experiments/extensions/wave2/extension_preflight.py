import argparse
from pathlib import Path
import subprocess
import sys
from metaexp.common import read,write,sha,code_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);a=p.parse_args();r=Path(__file__).resolve().parent
    parent=r.with_name(read(r/'campaign.json')['parent_campaign']);old_id=parent.name
    unit='metaexp-'+old_id+('-coordinator' if a.host=='pc2' else '-'+a.host)
    if subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()=='active':raise ValueError('Parent resource group still active')
    for args in [['-m','unittest','discover','-s','tests','-v'],['extension_tests.py']]:
        result=subprocess.run([sys.executable,*args],cwd=r,capture_output=True,text=True,timeout=180)
        (r/'preflight').mkdir(exist_ok=True);(r/'preflight'/('tests_'+a.host+'_'+str(len(args))+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError(result.stderr[-3000:])
    from metaexp.preflight import run
    proof=run(r,a.host,False);proof['original_acceptance_tests_passed']=True;proof['extension_guard_tests_passed']=True
    if a.host=='pc2':
        from metaexp.features import Bank
        from metaexp.reporting import verify_run
        smoke=r/'preflight/full_cv';smoke.mkdir(exist_ok=True)
        for name in ['bank','metaexp','.venv']:
            link=smoke/name
            if not link.exists():link.symlink_to(r/name,target_is_directory=True)
        recipe=next(v for v in read(r/'catalogue.json')['recipes'] if v['representation']['id']==read(r/'catalogue.json')['control'] and v['model']=='lgb')
        task={'id':'extension_cv_gate','mode':'cv','recipe':recipe,'device':'cpu'};write(smoke/'task.json',task)
        result=subprocess.run([sys.executable,str(r/'research_worker.py'),str(smoke),str(smoke/'task.json')],cwd=r,capture_output=True,text=True,timeout=180)
        (r/'preflight/full_cv.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('Guarded full CV failed: '+result.stderr[-1000:])
        score=verify_run(smoke,task['id'],Bank(r));proof['guarded_full_cv_ks']=score['pooled_oof']['ks'];proof['guarded_complete_three_fold_worker']=True
    write(r/'preflight'/f'{a.host}.json',proof)

if __name__=='__main__':main()
