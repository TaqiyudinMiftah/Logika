"""Refuse training unless the live hierarchical resource group is enforced."""
from pathlib import Path
import os,json,time
R=Path(__file__).resolve().parents[1]
line=[s for s in Path('/proc/self/cgroup').read_text().splitlines()if s.startswith('0::')][0]
cg=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
mem=(cg/'memory.max').read_text().strip();swap=(cg/'memory.swap.max').read_text().strip();cpu=(cg/'cpu.max').read_text().split()
passed=mem!='max'and int(mem)<=48*1024**3 and swap=='0'and cpu[0]!='max'and float(cpu[0])/int(cpu[1])<=8
(R/'RESOURCE_GATE.json').write_text(json.dumps(dict(status='passed'if passed else'failed',memory_max=mem,swap_max=swap,cpu_max=cpu,cgroup=str(cg),observed=time.time()),indent=2))
assert passed,'Live resource enforcement missing; no worker admitted'
os.execv(os.sys.executable,[os.sys.executable,str(R/'source/prepare.py')])
