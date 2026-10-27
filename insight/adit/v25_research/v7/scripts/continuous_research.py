"""Rolling, real-work queue with a minimum three-experiment research reserve."""
from pathlib import Path
import argparse,json,msvcrt,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path.insert(0,str(V/'scripts'))
from continue_frozen_kumo_queue import persistent_status


def definitions():
    files=json.loads((V/'CONTINUOUS_PLAN_FILES.json').read_text());rows=[]
    for name in files:
        assert Path(name).name==name and name.endswith('_plan.json')
        rows.extend(json.loads((V/name).read_text()))
    assert len({r['id'] for r in rows})==len(rows)
    return sorted([r for r in rows if r.get('research_ready')],key=lambda r:int(r['id'][1:]))


def state(stage,**extra):
    value=dict(status=stage,pid=os.getpid(),UTC=time.time(),minimum_pending_reserve=3,Kumo_frozen_fraction=1.,**extra)
    persistent_status(V/'reports/CONTINUOUS_QUEUE.json',value)
    print(json.dumps(value),flush=True)
    sys.path.insert(0,str(ROOT.parent))
    from node480_task import client,BASE
    # Status availability is useful but transient monitoring failure must not
    # duplicate or interrupt an already supervised real-model process.
    try:client().request('PUT','api/contents/'+BASE+'v7/reports/CONTINUOUS_QUEUE.json',dict(type='file',format='text',content=json.dumps(value,indent=2)))
    except Exception:pass


def main(expected):
    lock=(V/'reports/continuous_controller.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        if (V/'reports/CONTINUOUS_STOP_REQUEST.json').exists():
            state('STOP_REQUESTED_AT_EXPERIMENT_BOUNDARY',last_commit=expected);return
        rows=definitions();pending=[r for r in rows if not (V/'modeling/reports'/f"GITHUB_SYNC_{r['id']}.json").exists()]
        if len(pending)<=3:
            state('WAITING_FOR_RESEARCH_REFILL_RESERVE_PRESERVED',queued=[r['id'] for r in pending],queued_count=len(pending),last_commit=expected,note='Append additional evidence-based executable hypotheses; preserve three pending rather than emptying the queue.')
            time.sleep(15);continue
        spec=pending[0];remaining=[r['id'] for r in pending[1:]];assert len(remaining)>=3
        state('RUNNING_NEXT_EXPERIMENT',experiment=spec['id'],hypothesis=spec['rationale'],queued=remaining,queued_count=len(remaining),last_commit=expected)
        subprocess.run([sys.executable,str(V/'scripts/continue_frozen_kumo_queue.py'),'--expected-head',expected,'--runs',spec['id'],'--worker',spec['worker'],'--status-file','CONTINUOUS_CURRENT.json'],cwd=ROOT,check=True)
        receipt=json.loads((V/'modeling/reports'/f"GITHUB_SYNC_{spec['id']}.json").read_text());expected=receipt['commit']
        result=json.loads((V/'runs'/spec['id']/'RESULT.json').read_text());insight=result['continuous_insight']
        with (V/'reports/CONTINUOUS_RESEARCH_EVENTS.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(UTC=time.time(),commit=expected,**insight))+'\n')
        state('ANALYZED_AND_PUSHED',experiment=spec['id'],KS=result['OOF']['KS'],AUC=result['OOF']['AUC'],comparison_to_V193=result['comparison_to_V193'],findings=insight['findings'],queued=remaining,queued_count=len(remaining),last_commit=expected)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-head',required=True);args=p.parse_args()
    try:main(args.expected_head)
    except Exception:
        state('STOPPED_FOR_FAILURE_OR_PEER_REVIEW',traceback=traceback.format_exc());raise
