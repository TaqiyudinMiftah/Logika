"""Fetch exactly the unique public-candidate source files from completed inventories."""
import argparse
import collections
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private',required=True)
    parser.add_argument('--only',choices=['node2001','node2003','agung'])
    args=parser.parse_args();private=Path(args.private)
    known=set();sources={}
    for source in ['local','antigravity','node2001','node2003','agung']:
        if not (private/(source+'_inventory.jsonl')).exists():continue
        items=[json.loads(x)for x in (private/(source+'_inventory.jsonl')).open()]
        selected=[]
        for item in items:
            if item.get('category','').startswith('public_candidate') and item['sha256'] not in known:
                known.add(item['sha256'])
                if source.startswith('node') or source=='agung':selected.append(item)
        if source.startswith('node') or source=='agung':sources[source]=selected
    for source,items in sources.items():
        if args.only and source!=args.only:continue
        paths=[x['relative']for x in items]
        paths_set=set(paths)
        assert all(not Path(x).is_absolute() and '..'not in Path(x).parts for x in paths)
        out=private/'fetched'/source;out.mkdir(parents=True,exist_ok=True)
        listing=private/(source+'_fetch_paths.nul');listing.write_bytes(b'\0'.join(x.encode()for x in paths)+b'\0')
        if not paths:continue
        remote_base='/home/agung' if source=='agung'else '/mnt/home/faris'
        tar=['tar','-czf','-','--null','-C',remote_base,'-T','-']
        cmd=shlex.join(tar)
        if source=='node2003':cmd=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2003','faris@gpu-labkc.ub.ac.id',cmd])
        archive=private/(source+'_public_sources.tar.gz')
        print(json.dumps({'stage':'fetch','source':source,'unique_files':len(items),'bytes':sum(x['bytes']for x in items)}),flush=True)
        with listing.open('rb')as inp,archive.open('wb')as stream:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','agung@100.89.211.119' if source=='agung'else 'faris@100.68.29.126',cmd],stdin=inp,stdout=stream,check=True)
        with tarfile.open(archive,'r:gz')as tar:
            for member in tar.getmembers():
                assert member.isfile() and member.name in paths_set,member.name
            tar.extractall(out,filter='data')
        print(json.dumps({'stage':'fetched','source':source,'files':len(items),'archive_bytes':archive.stat().st_size}),flush=True)
