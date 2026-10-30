from pathlib import Path
import subprocess,time,json,concurrent.futures,re,os
os.environ["PATH"]="/home/faris/.local/bin:"+os.environ["PATH"]
R=Path('/home/faris/logika-twelvehour-cross-20261030')
def one(a):
 D=R/f'private/colab/account{a}';prefix=['/home/faris/.local/bin/colab','--account',str(a),'--config',str(D/'sessions.json')];v=dict(account=a,epoch=time.time());responses={}
 for op in ['usage','sessions']:
  p=subprocess.run(prefix+[op],text=True,capture_output=True,timeout=45);responses[op]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 (D/'TERMINAL_RESOURCE_PRIVATE.json').write_text(json.dumps(responses,indent=2)+'\n');s=responses['sessions'];u=responses['usage'];v['owned_absent']=s['returncode']==0 and f'logika-cross-a{a}-20261030' not in s['stdout'];v['assignment_count']=(re.search(r'Active assignments:\s*(\d+)',u['stdout'])or [None,None])[1];v['balance_CU']=(re.search(r'Current balance:\s*([\d.]+)',u['stdout'])or [None,None])[1];v['rate_CU_hour']=(re.search(r'Usage rate:\s*([\d.]+)',u['stdout'])or [None,None])[1];return v
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as ex:values=list(ex.map(one,range(1,12)))
v=dict(epoch=time.time(),accounts=values,all_owned_absent=all(x['owned_absent']for x in values),no_allocation_or_mutation=True);(R/'FINAL_RESOURCE_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
