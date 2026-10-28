"""Execution-only queue for fixed feature builds and inherited teacher refits."""
import argparse,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main(mode):
 for fold in range(5):
  assert not(R/'STOP').exists()
  if mode=='base':
   if(R/f'features/fold{fold}/BASE_MANIFEST.json').exists():continue
   script='build_production_features.py'
  else:
   while not(R/f'features/fold{fold}/BASE_MANIFEST.json').exists():
    assert not(R/'STOP').exists();time.sleep(2)
   if(R/f'features/fold{fold}/COMPLETE.json').exists():continue
   script='refit_feature_teachers.py'
  print(json.dumps({'stage':'dispatch','mode':mode,'fold':fold,'source':script}),flush=True)
  subprocess.run([sys.executable,str(R/'source'/script),'--fold',str(fold)],check=True)
 print(json.dumps({'stage':'queue_complete','mode':mode}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['base','teachers'],required=True);main(p.parse_args().mode)
