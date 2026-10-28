from pathlib import Path
import json,subprocess,tarfile,os
r=Path('/content/logika-all66');print('ENVIRONMENT',r.joinpath('ENVIRONMENT.json').read_text())
for name in ['pilot-run.log','production-run.log']:
 p=r/name
 if p.exists():print(name,'\n','\n'.join(p.read_text(errors='replace').splitlines()[-8:]))
complete=list((r/'pilot_outputs').glob('*/fold0/COMPLETE.json'));print('pilot_complete',len(complete))
if len(complete)==2:
 with tarfile.open(r/'pilot_results.tar.gz','w:gz')as tar:
  for name in ['pilot_outputs','ENVIRONMENT.json','PILOT_START.json','pilot-run.log']:tar.add(r/name,arcname=name)
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
