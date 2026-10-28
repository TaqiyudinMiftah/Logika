"""Start only with a newly supplied duration; preserve the clock on every resume."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from common import R,sha,dump,utc
import coordinator as c


def main(hours,resume=False):
    if resume:
        assert (R/'CLOCK.json').exists()and (R/'GATE.json').exists(),'No qualified campaign to resume'
        c.main();return
    assert hours is not None and 0<hours<=24,'Supply the newly authorized finite duration'
    assert not (R/'CLOCK.json').exists(),'Existing clock cannot be reset; use--resume after reconciling jobs'
    contract=json.loads((R/'VALIDATION_CONTRACT.json').read_text());contract['compute'].update(duration_status='New duration supplied for this invocation',ceiling_hours=hours)
    dump(contract,R/'VALIDATION_CONTRACT.json')
    now=time.time();clock={'started_UTC':utc(),'started_epoch':now,'search_deadline_epoch':now+hours*3600*.8,'deadline_epoch':now+hours*3600,
                         'hours':hours,'contract_sha256':sha(R/'VALIDATION_CONTRACT.json'),'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'configs_sha256':sha(R/'CONFIGS.json'),
                         'prior_campaign_clock_reused':False,'clock_policy':'New computation window includes gate/pilot/main/evaluation; research and implementation preparation recorded separately.'}
    dump(clock,R/'CLOCK.json');c.state('qualifying',clock=clock)
    subprocess.run([sys.executable,str(R/'source/gate.py')],check=True,timeout=180)
    nodes=json.loads((R/'TOPOLOGY.json').read_text())['nodes'];environments={n['name']:c.deploy(n)for n in nodes}
    assert len({v['tabm_source_sha256']for v in environments.values()})==1
    dump(environments,R/'ENVIRONMENTS.json');config=next(name for name,spec in json.loads((R/'CONFIGS.json').read_text()).items()if spec['kind']=='tabm'and spec['view']=='champion90'and spec['seed']==2026 and spec['loss']=='bce')
    job=None
    while job is None:
        assert time.time()<min(clock['search_deadline_epoch'],now+900),'No idleGPUavailable withinpilotadmissionlimit; preserve state and diagnose'
        for node in nodes:
            info=c.resources(node)
            for gpu,used,util in info['GPU']:
                if used<256 and util<20 and info['RAM_GiB']>=16:job=c.start(node,(config,0),'gpu',gpu,clock,pilot=True);break
            if job is not None:break
        if job is None:c.state('waiting_for_pilot_GPU',clock=clock);time.sleep(15)
    c.state('runtime_pilot',config=config,fold=0,node=job['node']['name'],slot=job['slot'])
    rc=job['process'].wait(timeout=660);job['log'].close();root=c.collect(job)
    assert rc==0,'Runtime pilot failed; broad admission blocked. Preserve pilot/logs and diagnose.'
    c.audit(root,0);pilot=json.loads((root/'RESULT.json').read_text());assert pilot['seconds']<600
    prep=json.loads((R/'GATE_PREP.json').read_text());gate={'status':'passed_for_declared_development_protocol','UTC':utc(),'contract_sha256':sha(R/'VALIDATION_CONTRACT.json'),
         'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'configs_sha256':sha(R/'CONFIGS.json'),'prepared_checks_sha256':sha(R/'GATE_PREP.json'),
         'runtime_pilot':{'config':config,'fold':0,'node':job['node']['name'],'seconds':pilot['seconds'],'result_path':str(root/'RESULT.json'),'result_sha256':sha(root/'RESULT.json'),'prediction_sha256':pilot['prediction_sha256']},
         'all_required_checks_passed':True,'independent_audit_available':False,'limitations':prep['limits']}
    dump(gate,R/'GATE.json');c.state('runtime_gate_passed',pilot=gate['runtime_pilot']);c.main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:main(a.hours,a.resume)
    except BaseException:c.state('failed',error=traceback.format_exc());raise
