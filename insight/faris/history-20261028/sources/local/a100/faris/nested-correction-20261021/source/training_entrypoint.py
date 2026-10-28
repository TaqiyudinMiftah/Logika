from pathlib import Path
import os,json,hashlib,time
R=Path(__file__).resolve().parents[1];p=json.loads((R/'training-plan.json').read_text())
for n,h in p['worker_source_sha256'].items():assert hashlib.sha256((R/'source'/n).read_bytes()).hexdigest()==h
line=[s for s in Path('/proc/self/cgroup').read_text().splitlines()if s.startswith('0::')][0];cg=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/');mem=(cg/'memory.max').read_text().strip();swap=(cg/'memory.swap.max').read_text().strip();cpu=(cg/'cpu.max').read_text().split();passed=mem!='max'and int(mem)<=32*1024**3 and swap=='0'and cpu[0]!='max'and float(cpu[0])/int(cpu[1])<=4
(R/'TRAINING_RESOURCE_GATE.json').write_text(json.dumps(dict(status='passed'if passed else'failed',memory_max=mem,swap_max=swap,cpu_max=cpu,cgroup=str(cg),observed=time.time()),indent=2));assert passed
os.execv(os.sys.executable,[os.sys.executable,str(R/'source/controller.py')])
