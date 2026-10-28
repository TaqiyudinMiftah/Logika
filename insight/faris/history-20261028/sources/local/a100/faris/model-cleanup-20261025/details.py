from pathlib import Path
import os,json,zipfile,tarfile,collections
ROOT=Path('/mnt/home/logika-faris-exploration');OUT=Path(__file__).resolve().parent
SKIP={'.git','.venv','venv','node_modules','__pycache__','.uv-cache','deps','pip-cache','site-packages'}
def main():
 rows=[];array_candidates=[];archive=[];headers=[]
 for base,ds,fs in os.walk(ROOT,followlinks=False):
  ds[:]=[n for n in ds if n not in SKIP and not Path(base,n).is_symlink()and not n.startswith('model-cleanup-')]
  for name in fs:
   p=Path(base,name)
   if p.is_symlink():continue
   rel=str(p.relative_to(ROOT));ext=p.suffix.lower()
   if ext in ['.joblib','.pkl','.pickle','.cbm','.pt','.pth','.ckpt','.safetensors','.onnx','.h5','.hdf5','.keras','.model','.bin','.ubj','.pb','.index']or'variables.data-'in name:
    rows.append({'path':rel,'bytes':p.stat().st_size,'extension':ext,'neighbors':sorted(x.name for x in p.parent.iterdir()if x.is_file()and x.suffix.lower()in ['.json','.yaml','.yml','.toml'])[:12]})
   if ext=='.npz':
    try:
     with zipfile.ZipFile(p)as z:keys=[x.filename for x in z.infolist()]
     if not all(any(s in x.lower()for s in ['prediction','prob','target','fold','rows','id','oof','seed','validation','train','test','baseline','control','spline','rank','true','shuff','both','ks','auc','class','bootstrap','context','correction','beta','maturity'])for x in keys):array_candidates.append({'path':rel,'keys':keys[:30]})
    except Exception as e:array_candidates.append({'path':rel,'error':str(e)})
   if (ext in ['.txt','.json','']or'/archive/objects/'in rel)and p.stat().st_size>100000:
    with p.open('rb')as f:h=f.read(160)
    if ext=='.txt'or'/archive/objects/'in rel or(h.startswith(b'{')and b'learner'in h):headers.append({'path':rel,'bytes':p.stat().st_size,'header':repr(h[:120])})
   if name.endswith(('.tar','.tar.gz','.tgz','.zip','.tar.xz','.tar.bz2')):
    try:
     if ext=='.zip':
      with zipfile.ZipFile(p)as z:members=[{'name':x.filename,'bytes':x.file_size}for x in z.infolist()]
     else:
      with tarfile.open(p,'r:*')as t:members=[{'name':x.name,'bytes':x.size}for x in t.getmembers()if x.isfile()]
     archive.append({'path':rel,'members':members})
    except Exception as e:archive.append({'path':rel,'error':str(e)})
 result={'models':rows,'array_review':array_candidates,'text_or_object_headers':headers,'archives':archive};(OUT/'details.json').write_text(json.dumps(result,indent=2));counts=collections.Counter(Path(x['path']).parts[0]for x in rows);print(json.dumps({'model_files':len(rows),'by_project':counts,'array_review_count':len(array_candidates),'header_count':len(headers),'archives':[{'path':x['path'],'members':len(x.get('members',[]))}for x in archive]},indent=2))
if __name__=='__main__':main()
