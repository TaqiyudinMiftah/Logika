from pathlib import Path
import os,json,collections,shutil,subprocess,datetime
ROOT=Path('/mnt/home/logika-faris-exploration')
def main():
 counts=collections.Counter();size=collections.Counter();examples={};big=[];archives=[];dirs=[];links=[]
 for base,ds,fs in os.walk(ROOT,followlinks=False):
  for n in list(ds):
   p=Path(base,n)
   if p.is_symlink():links.append({'path':str(p),'target':os.readlink(p)})
  ds[:]=[n for n in ds if n not in ['.git','.venv','venv','node_modules','__pycache__']and not Path(base,n).is_symlink()and not n.startswith('model-cleanup-')]
  for name in fs:
   p=Path(base,name)
   if p.is_symlink():
    links.append({'path':str(p),'target':os.readlink(p)});continue
   try:s=p.stat()
   except OSError:continue
   ext=p.suffix.lower()or'<none>';counts[ext]+=1;size[ext]+=s.st_size
   if len(examples.setdefault(ext,[]))<6:examples[ext].append(str(p.relative_to(ROOT)))
   if s.st_size>200*1024**2:big.append({'bytes':s.st_size,'path':str(p.relative_to(ROOT))})
   if name.endswith(('.tar','.tar.gz','.tgz','.zip','.tar.zst','.tar.xz','.tar.bz2')):archives.append({'bytes':s.st_size,'path':str(p.relative_to(ROOT))})
 result={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(ROOT),'free_GiB':shutil.disk_usage(ROOT).free/2**30,'extensions':[{'extension':e,'files':counts[e],'GiB':round(size[e]/2**30,4),'examples':examples[e]}for e in sorted(size,key=size.get,reverse=True)],'largest_files':sorted(big,key=lambda x:x['bytes'],reverse=True)[:65],'archives':archives,'symlinks':links,'owned_processes':subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,etimes,args'],text=True)}
 (Path(__file__).resolve().parent/'inventory.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='symlinks'},indent=2))
if __name__=='__main__':main()
