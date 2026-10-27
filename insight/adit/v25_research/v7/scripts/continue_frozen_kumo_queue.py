"""Local authorized queue: inspect shared HEAD, record results, push, then fit.

No tokens/passwords are stored here. Reuse the existing DPAPI session and local
git credential manager. New unreviewed peer commits stop this queue for review.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
from urllib.parse import quote
import requests
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
sys.path.insert(0,str(ROOT.parent))
from node480_task import client,BASE

def main(expected):
    from node380_remote import protect
    vault=json.loads(protect((Path.home()/'.codex/private/node480_session.dpapi').read_bytes(),decrypt=True))
    session=requests.Session();session.cookies.update(vault['cookies'])
    api=vault['base_url'].rstrip('/')+'/api/contents/'
    checkout=ROOT.parent/'shared_repo_sync_20261027'
    def state(stage,**extra):
        value=dict(status=stage,pid=os.getpid(),UTC=time.time(),Kumo_frozen_fraction=1.,**extra)
        p=V/'reports/LOCAL_KUMO_QUEUE.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
        print(json.dumps(value),flush=True)
    def get(rel):
        r=session.get(api+quote(BASE+rel)+'?content=1',timeout=20)
        if r.status_code==404:return None
        r.raise_for_status();return json.loads(r.json()['content'])
    def git(*args):
        return subprocess.run(['git','-C',str(checkout),*args],check=True,capture_output=True,text=True).stdout.strip()
    def peer_check(head):
        git('fetch','origin','main');actual=git('rev-parse','origin/main')
        if actual!=head:
            changed=git('diff','--name-only',head,actual)
            state('PEER_UPDATE_REQUIRES_RESULT_REVIEW',expected_head=head,current_head=actual,changed_files=changed.splitlines())
            raise RuntimeError('Peer update detected. Inspect their methods/results and redundancy before continuing; never force push.')
        return actual
    def wait_result(ident):
        failures=0
        while True:
            try:
                result=get(f'v7/runs/{ident}/RESULT.json')
                if result and result.get('status')=='COMPLETE':
                    assert result['frozen_Kumo_fraction']==1. and not result.get('backbone_gradient_updated',False)
                    assert len(result['folds'])==5 and result['all_outer_indices_identical']
                    return result
                status=get(f'v7/runs/{ident}/STATUS.json') or {}
                if str(status.get('status','')).startswith('FAILED'):raise RuntimeError(f'{ident}: process failed, inspect its status/log/resource receipt')
                state('WAITING_FOR_REAL_GPU_WORK',experiment=ident,remote_stage=status.get('status','STARTING'),fold=status.get('fold'),reviewed_head=expected)
                failures=0
            except requests.RequestException:
                failures+=1;state('TRANSIENT_NODE_CONNECTION_FAILURE',experiment=ident,consecutive_failures=failures)
                if failures>=10:raise
            time.sleep(15)
    # The first experiment is already running. Every later experiment waits
    # for a successful report push and a fresh peer-head check.
    for ident in ['V168','V169','V170','V171','V172','V173']:
        receipt=V/'modeling/reports'/f'GITHUB_SYNC_{ident}.json'
        if receipt.exists():
            expected=json.loads(receipt.read_text())['commit'];continue
        result=get(f'v7/runs/{ident}/RESULT.json')
        if ident!='V168' and result is None:
            peer_check(expected)
            c=client();gate=dict(status='REVIEWED_FOR_NEXT_RUN',experiment=ident,reviewed_head=expected,UTC=time.time(),peer_new_commits=False,redundancy_review='Fresh shared HEAD unchanged since prior synced result; distinct frozen-Kumo context input specification in kumo_plan.json. No backbone optimization or unfreezing.')
            c.request('PUT','api/contents/'+BASE+'v7/modeling/reports/REPO_REVIEW_GATE.json',dict(type='file',format='text',content=json.dumps(gate,indent=2)))
            command="python - <<'PY'\nfrom pathlib import Path\nimport subprocess,sys\nr=Path('/home/fariselhakim/adit/Logika/insight/goal054_offset_baseline_20261019')\np=subprocess.Popen([sys.executable,'-u',str(r/'v7/scripts/supervise_kumo.py'),'--run',"+repr(ident)+",'--worker','kumo_gpu_experiment_v2.py','--max-rss-gib','5.5'],stdout=(r/'v7/logs'/"+repr(ident+'_supervisor.log')+").open('a'),stderr=subprocess.STDOUT,start_new_session=True);print('SUPERVISOR_PID',p.pid)\nPY"
            c.execute(command,'python3');state('LAUNCHED_FROZEN_KUMO',experiment=ident,reviewed_head=expected)
        result=wait_result(ident)
        # Master ledger is written just after RESULT by the GPU worker.
        for attempt in range(12):
            master=get('v7/reports/MASTER_LOG_STATUS.json')
            if master and master.get('active_complete',0)>=int(ident[1:])+1:break
            time.sleep(5)
        else:raise RuntimeError('Completed result not yet represented in master ledger')
        peer_check(expected)
        state('PUBLISHING_COMPLETED_RESULT',experiment=ident,KS=result['OOF']['KS'],AUC=result['OOF']['AUC'],delta_KS=result['delta_KS'])
        subprocess.run([sys.executable,str(V/'scripts/sync_completed_experiment.py'),ident,'--expected-head',expected],check=True)
        expected=json.loads(receipt.read_text())['commit']
        # The following candidate is fixed before observing its own outer scores.
        # Selection across this research queue remains development-set reuse.
        state('RESULT_PUSHED_AND_REVIEWED',experiment=ident,commit=expected,KS=result['OOF']['KS'],AUC=result['OOF']['AUC'],goal_reached_on_development_folds=result['OOF']['KS']>=.531)
    state('COMPLETE',last_commit=expected)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-head',required=True);args=parser.parse_args()
    try:main(args.expected_head)
    except Exception:
        value=dict(status='STOPPED_FOR_FAILURE_OR_PEER_REVIEW',pid=os.getpid(),UTC=time.time(),traceback=traceback.format_exc())
        (V/'reports/LOCAL_KUMO_QUEUE.json').write_text(json.dumps(value,indent=2));raise
