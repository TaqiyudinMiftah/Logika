#!/usr/bin/python3
"""Noninteractive vipw editor: change exactly Faris's home field."""
from pathlib import Path
import os,sys
assert os.geteuid()==0
p=Path(sys.argv[1]);assert p.parent==Path('/etc')
original=p.read_text();lines=original.splitlines(keepends=True);matches=0
for i,line in enumerate(lines):
    if not line.startswith('faris:'):continue
    fields=line.rstrip('\n').split(':');assert len(fields)==7 and fields[2]=='1016'
    assert fields[5]in {'/home/faris','/mnt/home/faris'}
    fields[5]='/mnt/home/faris';lines[i]=':'.join(fields)+('\n'if line.endswith('\n')else'');matches+=1
assert matches==1
p.write_text(''.join(lines))
