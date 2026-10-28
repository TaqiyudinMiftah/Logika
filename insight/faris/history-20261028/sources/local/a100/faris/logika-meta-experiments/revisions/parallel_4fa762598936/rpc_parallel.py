"""Campaign-scoped batch transport; original worker and feature code stay frozen."""
import io
import gzip
import json
from pathlib import Path
import os
import shutil
import sys
import tarfile
import time
from metaexp.common import inside,read,write,sha

def folder_for(root,tid):
    if not isinstance(tid,str) or not tid or not all(c.isalnum() or c=='_' for c in tid):raise ValueError('Unsafe task ID')
    folder=inside(root,'runs/'+tid)
    if folder.stat().st_uid!=os.getuid():raise ValueError('Run is not owned by this account')
    return folder

def prioritize_gpu(root,tasks):
    # Queue metadata only: reserve the next free worker slot for an accelerator
    # task. The original daemon still enforces one GPU task and the worker caps.
    for task in tasks:
        if task.get('device') in ['cuda','opencl']:
            p=inside(root,'inbox/'+task['id']+'.json')
            if p.exists():os.utime(p,(1,1))

def pack_many(root,ids,stream):
    if not isinstance(ids,list) or not 1<=len(ids)<=32 or len(set(ids))!=len(ids):raise ValueError('Invalid result batch')
    folders=[folder_for(root,tid) for tid in ids]
    if any((f/'compacted.json').exists() for f in folders):raise ValueError('Result already archived on coordinator')
    with gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as tar:
            for folder in folders:
                for p in sorted(folder.rglob('*')):
                    if p.is_file() and not p.is_symlink():tar.add(p,arcname=str(p.relative_to(root)),recursive=False)

def compact(root,items):
    if not isinstance(items,list) or len(items)>32:raise ValueError('Invalid compaction batch')
    results=[]
    for item in items:
        folder=folder_for(root,item['id']);receipt=folder/'compacted.json'
        if sha(folder/'complete.json')!=item['complete_sha256']:raise ValueError('Central archive receipt does not match')
        if read(folder/'status.json')['state']!='completed':raise ValueError('Cannot compact an unfinished run')
        if receipt.exists():results.append({'id':item['id'],'already_archived':True});continue
        keep={'status.json','scores.json','identity.json','task.json','environment.json','resources.json','complete.json','compacted.json','worker.log'}
        write(receipt,{'time':time.time(),'complete_sha256':item['complete_sha256'],'authoritative_archive':'PC2:runs/'+item['id'],'mode':'truncate_payloads_preserving_directory_entries','reason':'Coordinator independently verified all transferred artifacts and OOF before requesting compaction.'})
        # The original daemon samples file sizes without a directory lock.
        # Preserve entries so compaction cannot race its is_file/stat sequence.
        for p in folder.rglob('*'):
            if p.parent==folder and p.name in keep:continue
            if p.is_file() and not p.is_symlink():
                with p.open('r+b') as f:f.truncate(0)
        log=folder/'worker.log'
        if log.exists() and log.stat().st_size>32768:
            with log.open('rb') as f:f.seek(-32768,2);tail=f.read()
            log.write_bytes(tail)
        results.append({'id':item['id'],'archived':True})
    return results

def main():
    root=Path(__file__).resolve().parent;raw=sys.stdin.read();request=json.loads(raw)
    if request['action']=='push':
        from metaexp.runtime import enqueue
        accepted=enqueue(root,request['tasks']);prioritize_gpu(root,request['tasks']);print(json.dumps({'accepted':accepted}),flush=True)
    elif request['action']=='pack_many':pack_many(root,request['ids'],sys.stdout.buffer)
    elif request['action']=='compact':print(json.dumps({'results':compact(root,request['items'])}),flush=True)
    else:
        from metaexp.runtime import rpc
        sys.stdin=io.StringIO(raw);rpc(root)

if __name__=='__main__':main()
