"""Explicit, scoped model cleanup. Never traverse symlink directories.

prepare freezes removal identities and hashes every retained research file;
export retains readable model settings without learned weight arrays;
apply only removes exported, identity-matched files; verify checks survivors.
"""
from pathlib import Path
import os, sys, json, hashlib, stat, time, collections, tarfile, zipfile
import re, pickle, io, traceback, functools, concurrent.futures, threading

ROOT=Path('/mnt/home/logika-faris-exploration')
OUT=ROOT/'model-cleanup-20261025'
SKIP={'.git','.venv','venv','node_modules','__pycache__','.uv-cache','deps','pip-cache','site-packages'}
MODEL_EXT={'.joblib','.cbm','.pt','.ckpt','.safetensors','.onnx','.h5','.hdf5','.keras','.model','.ubj'}
def dump(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,path)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def retained_evidence(p):
    """Full hashes for small research records; bounded samples for large data.

    Retained files are never opened for writing. Record inode, byte length and
    nanosecond timestamp for every file. Fully hash source and submission files;
    avoid scanning the dataset and tens of thousands of prediction arrays.
    """
    s=p.stat();row={'identity':ident(p)}
    if (p.suffix.lower()in {'.py','.md','.yaml','.yml','.toml','.ini','.cfg'} or (p.suffix.lower()=='.csv'and'submission'in p.name.lower()))and s.st_size<=8*1024*1024:
        row.update(hash_kind='full_sha256',sha256=sha(p))
    else:
        row.update(hash_kind='device_inode_size_mtime_ns',sha256=None)
    return row
def walk():
    for b,ds,fs in os.walk(ROOT,followlinks=False):
        ds[:]=[d for d in ds if d not in SKIP and not Path(b,d).is_symlink() and not d.startswith('model-cleanup-')]
        for f in fs:yield Path(b,f)
def reason(name,header=None):
    p=Path(name);n=p.name;e=p.suffix.lower()
    if e in MODEL_EXT:return 'serialized fitted model, transformer, or checkpoint'
    if e in {'.pkl','.pickle'}:
        if n in {'train.pkl','test.pkl','scheduler_rng.pkl'} or n.startswith('sampler_'):return None
        if '/models/'in '/'+name or n in {'fitted.pkl','fitted_context.pkl','state.pkl','quantile.pkl'}:return 'fitted context, transform or model state'
    if n in {'model.npz','context.npz'}and'/models/'in '/'+name:return 'formula coefficients or fitted training context'
    if n=='saved_model.pb'or n=='variables.index'or n.startswith('variables.data-'):return 'TensorFlow saved model'
    if e=='.txt'and header and header.startswith(b'tree\nversion='):return 'LightGBM tree model'
    if e=='.json'and header and header.startswith(b'{"learner":'):return 'XGBoost tree model'
    if '/archive/objects/'in '/'+name and header and (header.startswith(b'PK\x03\x04')or b'sdm.processing.'in header):return 'content-addressed model object'
    if '/blobs/'in '/'+name and header and header.startswith(b'PK\x03\x04')and b'data.pkl'in header:return 'Hugging Face cached Torch weights'
    return None
def ident(p):
    s=p.lstat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def header_for(p):
    # The initial content audit scanned every JSON file and found exactly the
    # three XGBoost k0/k1/k2 models. Other JSON records are preserved settings
    # and outputs, so do not reread tens of thousands on this spinning disk.
    if p.suffix.lower()=='.json'and p.name not in {'k0.json','k1.json','k2.json','model.json'}:return None
    if p.suffix.lower()in {'.txt','.json',''}or'/archive/objects/'in str(p):
        with p.open('rb')as f:return f.read(512)
    return None
def prepare():
    candidates=[];keep=[];links=[]
    for i,p in enumerate(walk()):
        rel=str(p.relative_to(ROOT))
        if p.is_symlink():
            links.append({'path':rel,'target':os.readlink(p),'identity':ident(p)});continue
        if not p.is_file():continue
        r=reason(rel,header_for(p))
        if r:candidates.append({'path':rel,'reason':r,'identity':ident(p),'bytes':p.stat().st_size})
        else:keep.append({'path':rel,'bytes':p.stat().st_size,**retained_evidence(p)})
        if i%5000==0:print('inventory',i,'models',len(candidates),'retained',len(keep),flush=True)
    planned={x['path']for x in candidates};remove_links=[]
    for l in links:
        p=ROOT/l['path']
        try:t=str(p.resolve().relative_to(ROOT))
        except (ValueError,RuntimeError):continue
        if t in planned:remove_links.append(l)
    archives=[]
    for a in json.load(open(OUT/'details.json'))['archives']:
        if a['path'].endswith('.zip'):continue
        if 'error'in a:continue
        p=ROOT/a['path'];removed=[]
        with tarfile.open(p,'r:*')as t:
            for m in t:
                if not m.isfile():continue
                f=t.extractfile(m);h=f.read(512);r=reason(m.name,h)
                if r:removed.append({'name':m.name,'bytes':m.size,'reason':r})
        if removed:archives.append({'path':a['path'],'sha256':sha(p),'identity':ident(p),'remove':removed})
    plan={'created_utc':time.strftime('%FT%TZ',time.gmtime()),'root':str(ROOT),'free_before':os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize,'models':candidates,'links':remove_links,'archives':archives,'retained_count':len(keep),'retained_bytes':sum(x['bytes']for x in keep)}
    dump(OUT/'removal-plan.json',plan)
    with(OUT/'retained-before.jsonl').open('w')as f:
        for k in keep:f.write(json.dumps(k)+'\n')
    print(json.dumps({'models':len(candidates),'logical_gib':sum(x['bytes']for x in candidates)/2**30,'model_links':len(remove_links),'archives_to_rewrite':len(archives),'retained_files':len(keep)}),flush=True)

class Placeholder:
    def __new__(cls,*a,**kw):return object.__new__(cls)
    def __init__(self,*a,**kw):
        if a:self._constructor_args=a
        if kw:self._constructor_kwargs=kw
    def __setstate__(self,s):
        if isinstance(s,dict):self.__dict__.update(s)
        else:self._state=s
    def __call__(self,*a,**kw):return Placeholder(*a,**kw)
    def append(self,x):self.__dict__.setdefault('_items',[]).append(x)
    def extend(self,x):self.__dict__.setdefault('_items',[]).extend(x)
    def __setitem__(self,k,v):self.__dict__.setdefault('_entries',{})[str(k)]=v
CLASSES={}
def find(module,name):
    if module.startswith(('numpy','joblib.numpy_pickle'))or module in {'builtins','collections','copyreg','_codecs'}:
        try:return getattr(__import__(module,fromlist=[name]),name)
        except ModuleNotFoundError:
            if module.startswith('numpy._core.'):return getattr(__import__(module.replace('numpy._core.','numpy.core.',1),fromlist=[name]),name)
            raise
    key=(module,name)
    if key not in CLASSES:CLASSES[key]=type(name,(Placeholder,),{'_original_class':module+'.'+name})
    return CLASSES[key]
def generic_load(p):
    # The historical interpreter was removed externally during this cleanup;
    # joblib itself is pure Python and also works with system Python's NumPy.
    try:import joblib
    except ModuleNotFoundError:sys.path.append(str(ROOT/'ranking-pipeline-20261020/.venv/lib/python3.11/site-packages'))
    from joblib.numpy_pickle import NumpyUnpickler,_validate_fileobject_and_memmap
    class Reader(NumpyUnpickler):
        def find_class(self,m,n):return find(m,n)
        def persistent_load(self,pid):return Placeholder(persistent_id=pid)
    with open(p,'rb')as f:
        with _validate_fileobject_and_memmap(f,str(p),None)as (h,mm):
            return Reader(str(p),h,ensure_native_byte_order=True,mmap_mode=None).load()
def generic_bytes(b):
    class Reader(pickle.Unpickler):
        def find_class(self,m,n):return find(m,n)
        def persistent_load(self,pid):return Placeholder(persistent_id=pid)
    return Reader(io.BytesIO(b)).load()
DROP={'model_str','model_string','state_dict','model_state_dict','optimizer','optimizer_state_dict','weights','estimators_','tree_','trees_','context','bank','X','x','y','train_x','train_y','_Booster','_object','_constructor_args'}
def clean(v,key='',depth=0,seen=None):
    import numpy as np
    if seen is None:seen=set()
    if v is None or isinstance(v,(str,bool,int,float)):
        if isinstance(v,float)and not np.isfinite(v):return str(v)
        if isinstance(v,str)and len(v)>20000:return {'text_length':len(v),'sha256':hashlib.sha256(v.encode()).hexdigest()}
        return v
    if isinstance(v,np.generic):return clean(v.item(),key,depth,seen)
    if isinstance(v,np.ndarray):
        if v.size<=4096 and (key in {'coefficients','coef_','intercept','intercept_','indices','selected','columns','heads','feature_names_in_','classes_'}):return clean(v.tolist(),key,depth+1,seen)
        return {'array_shape':list(v.shape),'dtype':str(v.dtype),'learned_array_not_copied':True}
    if isinstance(v,bytes):return {'bytes':len(v)}
    if id(v)in seen:return {'reference':type(v).__name__}
    if depth>12:return {'class':getattr(type(v),'_original_class',type(v).__name__),'depth_limit':True}
    seen.add(id(v))
    if isinstance(v,dict):
        return {str(k):({'excluded_learned_state':True}if str(k)in DROP else clean(x,str(k),depth+1,seen))for k,x in list(v.items())[:10000]}
    if isinstance(v,(tuple,list,set)):
        if len(v)>4096:return {'length':len(v),'type':type(v).__name__}
        return [clean(x,key,depth+1,seen)for x in v]
    if hasattr(v,'__dict__'):
        return {'class':getattr(type(v),'_original_class',type(v).__module__+'.'+type(v).__name__),'settings':clean(vars(v),key,depth+1,seen)}
    return {'class':type(v).__module__+'.'+type(v).__name__}
@functools.lru_cache(maxsize=100000)
def configs_in(parent):
    return tuple(str(q.relative_to(ROOT))for q in parent.iterdir()if q.is_file()and not q.is_symlink()and q.suffix.lower()in {'.json','.yaml','.yml','.toml','.ini','.cfg'}and q.name not in {'k0.json','k1.json','k2.json'})
def config_refs(p):
    refs=[];parent=p.parent
    for _ in range(5):
        if parent==ROOT:break
        refs.extend(configs_in(parent))
        parent=parent.parent
    return sorted(set(refs))
def metadata(p):
    result={'configuration_files_preserved':config_refs(p)};e=p.suffix.lower()
    if e in {'.joblib','.pkl','.pickle'}or('/archive/objects/'in str(p)and not zipfile.is_zipfile(p)):
        # Placeholder classes retain configuration state without importing training code.
        result['embedded_settings']=clean(generic_load(p))
    elif e=='.npz':
        import numpy as np
        with np.load(p,allow_pickle=False)as z:result['embedded_settings']={k:clean(z[k],k)for k in z.files}
    elif e=='.cbm':
        # CatBoost stores a JSON configuration in the FlatBuffer metadata prefix.
        with p.open('rb')as f:b=f.read(8*1024*1024)
        txt=b.decode('utf8','replace');found=[];decoder=json.JSONDecoder()
        for m in re.finditer(r'\{"(?:model_based_eval_options|detailed_profile|boosting_options|flat_params|iterations|depth|loss_function)"',txt):
            try:v,_=decoder.raw_decode(txt[m.start():])
            except ValueError:continue
            if isinstance(v,dict):found.append(v)
        result['catboost_metadata']=found
        if not found:result['embedded_metadata_notice']='No configuration decoded from bounded prefix; preserved configuration references and source remain.'
    elif e=='.txt':
        with p.open('rb')as f:
            first=f.read(512*1024).decode('utf8','replace').split('Tree=0')[0]
            f.seek(max(0,p.stat().st_size-512*1024));tail=f.read().decode('utf8','replace')
        result['lightgbm_header']=first
        if 'parameters:'in tail:result['lightgbm_parameters']=tail.split('parameters:',1)[1].split('end of parameters',1)[0]
    elif e=='.json':
        v=json.load(open(p));learner=v.get('learner',{});result['xgboost_settings']={k:v for k,v in learner.items()if k!='gradient_booster'}
        gb=learner.get('gradient_booster',{});result['xgboost_booster_type']=gb.get('name');result['xgboost_tree_configuration']=gb.get('model',{}).get('gbtree_model_param',{})
    elif zipfile.is_zipfile(p):
        with zipfile.ZipFile(p)as z:
            if e=='.keras':result['keras_configuration']={n:json.loads(z.read(n))for n in z.namelist()if n in {'config.json','metadata.json'}}
            names=[n for n in z.namelist()if n.endswith('/data.pkl')or n=='data.pkl']
            if names:result['embedded_settings']=clean(generic_bytes(z.read(names[0])))
            else:result['archive_members']=[{'name':x.filename,'bytes':x.file_size}for x in z.infolist()]
    else:result['embedded_metadata_notice']='Binary model format; configuration files and training source preserved.'
    return result
def export(limit=None,direct=False):
    if direct:
        d=json.load(open(OUT/'details.json'));names=set(x['path']for group in ['models','text_or_object_headers','array_review']for x in d[group]);rows=[]
        for name in sorted(names):
            p=ROOT/name
            if p.is_symlink()or not p.is_file():continue
            r=reason(name,header_for(p))
            if r:rows.append({'path':name,'bytes':p.stat().st_size,'identity':ident(p),'reason':r})
        plan={'models':rows};dump(OUT/'export-candidates.json',plan)
    else:plan=json.load(open(OUT/'removal-plan.json'))
    dest=OUT/'parameters';dest.mkdir(exist_ok=True)
    errors=[];n=0
    for row in plan['models']:
        p=ROOT/row['path'];k=hashlib.sha256(row['path'].encode()).hexdigest();out=dest/(k+'.json')
        if out.exists():continue
        if ident(p)!=row['identity']:raise RuntimeError('Model changed: '+str(p))
        record={'path':row['path'],'bytes':row['bytes'],'format':p.suffix,'exported_utc':time.strftime('%FT%TZ',time.gmtime())}
        try:record.update(metadata(p))
        except Exception as e:
            record['embedded_export_error']=type(e).__name__+': '+str(e)[:1000];record['configuration_files_preserved']=config_refs(p);errors.append({'path':row['path'],'error':record['embedded_export_error']})
        dump(out,record);n+=1
        if n%200==0:print('exported',n,'errors',len(errors),flush=True)
        if limit and n>=limit:break
    dump(OUT/'export-errors.json',errors)
    print('exports_written',n,'errors',len(errors),flush=True)
def parallel_export():
    """Four I/O workers; only one large deserialization may run at a time."""
    plan=json.load(open(OUT/('removal-plan.json'if(OUT/'removal-plan.json').exists()else'export-candidates.json')))
    dest=OUT/'parameters';dest.mkdir(exist_ok=True);large=threading.Lock()
    def one(row):
        p=ROOT/row['path'];out=dest/(hashlib.sha256(row['path'].encode()).hexdigest()+'.json')
        if out.exists():return False
        if ident(p)!=row['identity']:raise RuntimeError('Model changed: '+str(p))
        record={'path':row['path'],'bytes':row['bytes'],'format':p.suffix}
        try:
            if row['bytes']>128*1024*1024:
                with large:record.update(metadata(p))
            else:record.update(metadata(p))
        except Exception as e:
            record['embedded_export_error']=type(e).__name__+': '+str(e)[:1000];record['configuration_files_preserved']=config_refs(p)
        dump(out,record);return True
    n=0
    pending=[r for r in plan['models']if not(dest/(hashlib.sha256(r['path'].encode()).hexdigest()+'.json')).exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for done in pool.map(one,pending):
            n+=int(done)
            if n and n%500==0:print('new exports',n,'of',len(pending),flush=True)
    errors=[]
    for p in dest.glob('*.json'):
        d=json.load(open(p))
        if 'embedded_export_error'in d:errors.append({'path':d.get('path',d.get('member')),'error':d['embedded_export_error']})
    dump(OUT/'export-errors.json',errors);print('export complete; errors',len(errors),flush=True)
def repair_exports():
    """Retry failed historical-interpreter exports using working system Python."""
    plan=json.load(open(OUT/'removal-plan.json'));dest=OUT/'parameters';pending=[];errors=[];large=threading.Lock()
    for r in plan['models']:
        out=dest/(hashlib.sha256(r['path'].encode()).hexdigest()+'.json')
        if not out.exists()or'embedded_export_error'in json.load(open(out)):pending.append(r)
    print('retry/missing metadata exports',len(pending),flush=True)
    def one(r):
        p=ROOT/r['path'];record={'path':r['path'],'bytes':r['bytes'],'format':p.suffix,'metadata_runtime':'system Python'}
        try:
            if r['bytes']>128*1024*1024:
                with large:record.update(metadata(p))
            else:record.update(metadata(p))
        except Exception as e:record['embedded_export_error']=type(e).__name__+': '+str(e);record['configuration_files_preserved']=config_refs(p)
        dump(dest/(hashlib.sha256(r['path'].encode()).hexdigest()+'.json'),record)
        return {'path':r['path'],'error':record['embedded_export_error']}if'embedded_export_error'in record else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for i,result in enumerate(pool.map(one,pending),1):
            if result:errors.append(result)
            if i%500==0:print('retried',i,'errors',len(errors),flush=True)
    dump(OUT/'export-errors.json',errors);print('metadata repair finished; errors',len(errors),flush=True)
def active():
    hits=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=str((proc/'cwd').resolve())
        except OSError:continue
        if str(OUT)in cmd:continue
        if str(ROOT)in cmd or cwd.startswith(str(ROOT)):
            # Only training/interpreters using this root need hold the cleanup.
            if any(s in cmd for s in ['python','train','controller','worker']):hits.append({'pid':proc.name,'command':cmd[:600],'cwd':cwd})
    return hits
def supplement():
    """Resolve extensionless HF blobs and any last format matches before apply."""
    plan=json.load(open(OUT/'removal-plan.json'));known={x['path']for x in plan['models']};links=[]
    for p in walk():
        rel=str(p.relative_to(ROOT))
        if p.is_symlink():links.append(p);continue
        if not p.is_file()or rel in known:continue
        r=reason(rel,header_for(p))
        if r:plan['models'].append({'path':rel,'bytes':p.stat().st_size,'identity':ident(p),'reason':r});known.add(rel)
    plan['links']=[]
    for p in links:
        try:t=str(p.resolve().relative_to(ROOT))
        except(ValueError,RuntimeError):continue
        if t in known or reason(str(p.relative_to(ROOT))):plan['links'].append({'path':str(p.relative_to(ROOT)),'target':os.readlink(p),'identity':ident(p)})
    records=[json.loads(l)for l in open(OUT/'retained-before.jsonl')]
    records=[r for r in records if r['path']not in known]
    with(OUT/'retained-before.jsonl').open('w')as f:
        for r in records:f.write(json.dumps(r)+'\n')
    plan.update(retained_count=len(records),retained_bytes=sum(r['bytes']for r in records));dump(OUT/'removal-plan.json',plan)
    # Refresh Keras exports with its embedded architecture/training JSON.
    for r in plan['models']:
        if r['path'].endswith('.keras'):
            p=ROOT/r['path'];record={'path':r['path'],'bytes':r['bytes'],'format':p.suffix,**metadata(p)}
            dump(OUT/'parameters'/(hashlib.sha256(r['path'].encode()).hexdigest()+'.json'),record)
    print('final plan models',len(plan['models']),'links',len(plan['links']),flush=True)
def rewrite(a):
    p=ROOT/a['path'];tmp=p.with_name(p.name+'.cleanup-tmp');removed={x['name']for x in a['remove']}
    if ident(p)!=a['identity']or sha(p)!=a['sha256']:raise RuntimeError('Archive changed: '+str(p))
    retained=[];exports=[]
    with tarfile.open(p,'r:*')as src,tarfile.open(tmp,'w:gz',compresslevel=3)as dst:
        for m in src:
            if m.name in removed:
                # Each stripped member has an unpacked counterpart already exported.
                tempdir=OUT/'archive-member-staging';tempdir.mkdir(exist_ok=True)
                staged=tempdir/Path(m.name).name
                with staged.open('wb')as f:
                    source=src.extractfile(m)
                    for b in iter(lambda:source.read(8*1024*1024),b''):f.write(b)
                member_record={'archive':a['path'],'member':m.name,'bytes':m.size}
                try:member_record.update(metadata(staged))
                except Exception as e:member_record['embedded_export_error']=type(e).__name__+': '+str(e)
                member_record['configuration_files_preserved']='All non-model archive members retained and individually verified in archive receipt.'
                paramfile=OUT/'parameters'/('archive-'+hashlib.sha256((a['path']+'!'+m.name).encode()).hexdigest()+'.json')
                dump(paramfile,member_record);staged.unlink()
                exports.append({'name':m.name,'bytes':m.size,'parameter_export':paramfile.name});continue
            if m.isfile():
                f=src.extractfile(m);b=f.read();retained.append({'name':m.name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
                dst.addfile(m,io.BytesIO(b))
            else:dst.addfile(m)
    with tarfile.open(tmp,'r:*')as check:
        observed=[]
        for m in check:
            if m.isfile():observed.append({'name':m.name,'sha256':hashlib.sha256(check.extractfile(m).read()).hexdigest(),'bytes':m.size})
    if observed!=retained:raise RuntimeError('Archive member mismatch')
    os.chmod(tmp,stat.S_IMODE(p.stat().st_mode));os.replace(tmp,p)
    rec={'path':a['path'],'old_sha256':a['sha256'],'new_sha256':sha(p),'removed_members':exports,'retained_members':retained}
    dump(OUT/('archive-'+hashlib.sha256(a['path'].encode()).hexdigest()+'.json'),rec)
    return rec
def apply():
    plan=json.load(open(OUT/'removal-plan.json'));hits=active()
    if hits:dump(OUT/'active-blockers.json',hits);raise RuntimeError('Active research process; cleanup stopped')
    for r in plan['models']:
        if not(OUT/'parameters'/(hashlib.sha256(r['path'].encode()).hexdigest()+'.json')).exists():raise RuntimeError('Missing parameters export: '+r['path'])
        p=ROOT/r['path']
        if p.exists()and ident(p)!=r['identity']:raise RuntimeError('File changed: '+r['path'])
    # Archive edits first: failure cannot discard retained outputs.
    for a in plan['archives']:
        receipt=OUT/('archive-'+hashlib.sha256(a['path'].encode()).hexdigest()+'.json')
        if not receipt.exists():rewrite(a)
        print('archive preserved:',a['path'],flush=True)
    n=0
    with(OUT/'deletions.jsonl').open('a')as ledger:
        for r in plan['models']:
            p=ROOT/r['path']
            if not p.exists():continue
            if p.is_symlink()or ident(p)!=r['identity']or not p.resolve().is_relative_to(ROOT):raise RuntimeError('Identity/containment changed: '+r['path'])
            s=p.stat();blocks=s.st_blocks*512;p.unlink();ledger.write(json.dumps({**r,'allocated_bytes_before':blocks,'hard_links_before':s.st_nlink,'released_allocated_bytes':blocks if s.st_nlink==1 else 0,'deleted_utc':time.strftime('%FT%TZ',time.gmtime())})+'\n');n+=1
            if n%500==0:ledger.flush();os.fsync(ledger.fileno());print('deleted',n,flush=True)
        ledger.flush();os.fsync(ledger.fileno())
    for l in plan['links']:
        p=ROOT/l['path']
        if p.is_symlink()and ident(p)==l['identity']:p.unlink()
    print('removed files:',n,flush=True)
def verify():
    plan=json.load(open(OUT/'removal-plan.json'));changed_archives={a['path']:json.load(open(OUT/('archive-'+hashlib.sha256(a['path'].encode()).hexdigest()+'.json')))for a in plan['archives']}
    errors=[];checked=0;kept_bytes=0;hash_kinds=collections.Counter()
    for line in open(OUT/'retained-before.jsonl'):
        r=json.loads(line);p=ROOT/r['path'];hash_kinds[r['hash_kind']]+=1
        if not p.is_file():errors.append({'path':r['path'],'reason':'retained file missing'})
        elif r['path']in changed_archives:
            if sha(p)!=changed_archives[r['path']]['new_sha256']:errors.append({'path':r['path'],'reason':'rewritten archive mismatch'})
        elif retained_evidence(p)!= {k:r[k]for k in ['identity','hash_kind','sha256']}:
            errors.append({'path':r['path'],'reason':'retained file identity or content evidence changed'})
        checked+=1;kept_bytes+=r['bytes']
        if checked%10000==0:print('verified retained',checked,flush=True)
    remaining=[]
    for p in walk():
        if p.is_symlink()or not p.is_file():continue
        r=reason(str(p.relative_to(ROOT)),header_for(p))
        if r:remaining.append(str(p.relative_to(ROOT)))
    free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
    result={'finished_utc':time.strftime('%FT%TZ',time.gmtime()),'root':str(ROOT),'deleted_model_files':len(plan['models'])-len([r for r in plan['models']if(ROOT/r['path']).exists()]),'deleted_model_links':len(plan['links']),'removed_archive_model_members':sum(len(a['remove'])for a in plan['archives']),'retained_files_verified':checked,'retained_bytes_before':kept_bytes,'retained_hash_failures':errors,'remaining_model_files':remaining,'free_before_bytes':plan['free_before'],'free_after_bytes':free,'net_freed_gib':(free-plan['free_before'])/2**30,'parameter_exports':len(list((OUT/'parameters').glob('*.json'))),'embedded_export_errors':len(json.load(open(OUT/'export-errors.json')))}
    result['retained_verification_methods']=dict(hash_kinds)
    result['model_allocated_gib_unlinked_at_last_hardlink']=sum(json.loads(l).get('released_allocated_bytes',0)for l in open(OUT/'deletions.jsonl'))/2**30
    result['archive_parameter_export_errors']=[{'file':p.name,'error':d['embedded_export_error']}for p in(OUT/'parameters').glob('archive-*.json')for d in[json.load(open(p))]if'embedded_export_error'in d]
    result['success']=not errors and not remaining
    dump(OUT/'REPORT.json',result);print(json.dumps(result,indent=2),flush=True)
    if not result['success']:sys.exit(1)
if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    cmd=sys.argv[1]
    if cmd=='prepare':prepare()
    elif cmd=='export':export(int(sys.argv[2])if len(sys.argv)>2 else None)
    elif cmd=='export-direct':export(direct=True)
    elif cmd=='export-parallel':parallel_export()
    elif cmd=='repair-exports':repair_exports()
    elif cmd=='apply':apply()
    elif cmd=='verify':verify()
    elif cmd=='active':print(json.dumps(active(),indent=2))
    elif cmd=='supplement':supplement()
