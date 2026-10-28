"""Append a hash-audited supplement without rewriting historical archive seals.

Every discovered file gets a publication/alias/private reference. Sources remain
unchanged; only code, aggregate evidence and metadata can enter this public repo.
"""
import argparse,collections,gzip,hashlib,json,subprocess
from pathlib import Path
from publish_experiment_history import public_content,clean_json

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def reconcile(repo,sources,out,known_trees=()):
 base=repo/'insight/faris/history-20261028';index=json.loads((base/'SOURCE_INVENTORY_INDEX.json').read_text());known={};public={}
 for part in index['parts']:
  p=base/part['file'];assert sha(p)==part['sha256'],p
  for line in gzip.open(p,'rt'):
   r=json.loads(line)
   if r.get('publication')in{'archived','deduplicated_alias'}:
    location={k:r[k]for k in ['public_path','public_archive','archive_member','public_sha256']if k in r}
    known.setdefault(r['sha256'],location);public.setdefault(r['public_sha256'],location)
 # Completed continuations can already be published outside the old archive.
 # Only tracked files qualify; the supplement must never alias its own output.
 if known_trees:
  tracked=subprocess.check_output(['git','-C',str(repo),'ls-files','-z','--',*known_trees]).decode().split('\0')
  for relative in filter(None,tracked):
   p=repo/relative;digest=sha(p)
   location={'public_path':relative,'public_sha256':digest}
   known.setdefault(digest,location);public.setdefault(digest,location)
 raw={}
 for name,inv,fetch in sources:
  for line in inv.open():
   r=json.loads(line)
   if r.get('category','').startswith('public')and r.get('sha256'):
    p=Path(r['path'])if name=='local'else fetch/r['relative']
    if p.exists():raw.setdefault(r['sha256'],p)
 out.mkdir(parents=True,exist_ok=True);records=[];seen=set();counts=collections.Counter();transforms=collections.Counter();bytes_public=0;new=[]
 for name,inv,fetch in sources:
  for line in inv.open():
   item=json.loads(line);key=(name,item['path'])
   if key in seen:continue
   seen.add(key);r={'source':name,**item}
   if item['kind']!='file':records.append(r);continue
   if not item['category'].startswith('public'):
    r['publication']='private_reference'
   elif item['sha256']in known:
    r.update(known[item['sha256']]);r['publication']='deduplicated_alias';r['transformation']='Original-content SHA256 matches an already published source; existing privacy transformation retained.'
   else:
    p=raw.get(item['sha256']);assert p is not None,('Missing source bytes',name,item['path'])
    assert sha(p)==item['sha256'],('Source changed since inventory',p)
    content,policy=public_content(p,Path(item['path']).name);r['transformation']=policy;transforms[policy]+=1
    if content is None:r['publication']='private_reference'
    else:
     digest=hashlib.sha256(content).hexdigest()
     if digest in public:r.update(public[digest]);r['publication']='deduplicated_alias'
     else:
      rel=Path(item['relative']);assert not rel.is_absolute()and'..'not in rel.parts
      target=out/'sources'/name/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
      r.update(publication='archived',public_path=str(target.relative_to(repo)),public_sha256=digest,public_bytes=len(content));bytes_public+=len(content);new.append({'source':name,'source_path':item['path'],'public_path':r['public_path'],'public_sha256':digest})
      public[digest]={k:r[k]for k in ['public_path','public_sha256']}
     known[item['sha256']]={k:r[k]for k in ['public_path','public_archive','archive_member','public_sha256']if k in r}
   counts[r['publication']]+=1;records.append(r)
 manifest=out/'SOURCE_INVENTORY.jsonl.gz'
 with manifest.open('wb')as target,gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0)as stream:
  for r in records:stream.write((json.dumps(clean_json(r),ensure_ascii=False)+'\n').encode())
 # The supplement must neither hide scan errors nor claim unavailable bytes exist.
 assert not any(r['kind']in{'scan_error','missing_root'}for r in records)
 for r in records:
  if r.get('publication')=='archived':assert sha(repo/r['public_path'])==r['public_sha256']
 inputs=[{'source':n,'inventory_sha256':sha(i),'records':sum(1 for _ in i.open())}for n,i,_ in sources]
 result={'status':'HASH_VERIFIED_PENDING_SECRET_SCAN_AND_PUSH','records':len(records),'input_inventories':inputs,'known_tracked_trees':list(known_trees),'publication_counts':dict(counts),'new_public_bytes':bytes_public,'transformations':dict(transforms),'inventory_sha256':sha(manifest),'new_public_files':new,'limits':['Source records are files/aliases/private references, not independent experiment counts.','Retired node2002 was not accessed. Raw data, case-level labels/predictions and fitted weights remain private.','Historical import verification is preserved; this supplement corrects its documented discovery gaps.']}
 (out/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='new_public_files'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True);p.add_argument('--known-tree',action='append',default=[]);p.add_argument('--source',action='append',nargs=3,required=True);a=p.parse_args();reconcile(Path(a.repo),[(n,Path(i),Path(f))for n,i,f in a.source],Path(a.out),a.known_tree)
