"""User-invoked campaign operations. No polling loops or automation registration."""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logika-meta-experiments'));from metaexp.deployment import remote
PC2='/mnt/home/logika-faris-exploration/'+ROOT.name
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','launch','status','stop','resume','collect','report','verify','replay']);p.add_argument('host',nargs='?',default='all',choices=['all','pc1','pc2','pc3','node380']);p.add_argument('--job');p.add_argument('--output');a=p.parse_args()
args=[a.action,a.host]
if a.job:args+=['--job',a.job]
if a.output:args+=['--output',a.output]
cmd=PC2+'/.venv/bin/python '+PC2+'/operate.py '+shlex.join(args)
x=remote('pc2',cmd,capture_output=True,text=True,timeout=55);print(x.stdout)
