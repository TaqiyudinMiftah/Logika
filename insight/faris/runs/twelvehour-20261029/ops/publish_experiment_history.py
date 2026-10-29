"""Sanitize and deduplicate historical experiment records for this public repository.

Raw data, case-level labels/OOF, environments and fitted weights remain private.
Their inventory references are retained; public record counts are not independent
experiment counts. Originals are never modified.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

SECRET_KEYS = {'password','passwd','pass','api_key','apikey','access_token','refresh_token',
               'id_token','client_secret','authorization','cookie','kaggle_key','secret','token'}
PATTERNS = [
    re.compile(r'KGAT_[A-Za-z0-9_-]{12,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(r'hf_[A-Za-z0-9]{20,}'),
    re.compile(r'AIza[A-Za-z0-9_-]{25,}'),
    re.compile(r'ya29\.[A-Za-z0-9_.-]{20,}'),
    re.compile(r'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', re.S),
]
CASE_KEYS = {'application_id','customer_id','target','label','prediction','probability','proba','oof','y_true','y_pred'}
CASE_ARRAY_KEYS = {'target','targets','labels','y','y_true','y_pred','prediction','predictions','proba','raw','oof','baseline','validation_predictions'}
CASE_ID_KEYS = {'application_id','customer_id','sk_id_curr','row_index','row_id','id'}
CASE_VALUE_KEYS = {'target','label','y','y_true','y_pred','prediction','predictions','probability','proba','prob','pred','score'}
DIRECT_CASE_ARRAY_KEYS = {'target','targets','labels','y','y_true','y_pred','prediction','predictions',
                          'probability','proba','prob','pred','oof','validation_predictions',
                          'application_id','customer_id','sk_id_curr','row_index','row_id'}


def sanitize(text):
    low=text.lower()
    if not any(word in low for word in ['kgat_','github_pat_','ghp_','gho_','ghu_','ghs_','ghr_','hf_','aiza','ya29.','sk-','private key','token','bearer','password','passwd','sshpass','sudo','api_key','client_secret','pass=','pass =']):
        return text
    for pattern in PATTERNS:
        text = pattern.sub('[REDACTED_CREDENTIAL]', text)
    text = re.sub(r'(?i)([?&](?:token|access_token|api_key)=)[^\s&"\'<>]+',r'\1[REDACTED_CREDENTIAL]',text)
    text = re.sub(r'(?i)(Bearer\s+)[A-Za-z0-9._-]{18,}',r'\1[REDACTED_CREDENTIAL]',text)
    text = re.sub(r'''(?i)(["']?(?:api_key|access_token|refresh_token|client_secret|token|password|passwd)["']?\s*[:=]\s*["'])([A-Za-z0-9_.-]{6,})(["'])''',r'\1[REDACTED_CREDENTIAL]\3',text)
    text = re.sub(r'(?i)(sshpass\s+-p\s+)(["\']?)[^\s"\']+\2',r'\1[REDACTED_CREDENTIAL]',text)
    text = re.sub(r'(?i)((?:KAGGLE_API_TOKEN|HF_TOKEN|GITHUB_TOKEN|OPENAI_API_KEY|GOOGLE_API_KEY|JUPYTER_TOKEN|PASSWORD|PASS)\s*[:=]\s*["\']?)([A-Za-z0-9_-]{6,})(["\']?)',r'\1[REDACTED_CREDENTIAL]\3',text)
    text = re.sub(r'(?i)((?:echo|printf)\s+["\']?)(\d{6,})(["\']?\s*\|\s*sudo)',r'\1[REDACTED_CREDENTIAL]\3',text)
    return text


def clean_json(value, key=''):
    if isinstance(value, dict):
        normalized={str(k).lower():v for k,v in value.items()}
        row_id=normalized.get('id')
        identifies_case=bool((CASE_ID_KEYS-{'id'}) & set(normalized)) or isinstance(row_id,(int,float)) or isinstance(row_id,str) and row_id.isdigit()
        if identifies_case and any(isinstance(normalized.get(k),(int,float,bool))for k in CASE_VALUE_KEYS):
            return {'private_case_row_omitted':True}
        if len(value)>1000 and all(str(k).isdigit() for k in value):
            return {'private_case_map_omitted':len(value)}
        return {k:('[REDACTED_CREDENTIAL]' if (k.lower() in SECRET_KEYS or k.lower()=='key' and 'username'in normalized) and isinstance(v,str) and v else clean_json(v,k.lower())) for k,v in value.items()}
    if isinstance(value,list):
        if len(value)>100 and all(isinstance(v,list) and v and all(x is None or isinstance(x,(int,float,bool))for x in v)for v in value):
            return {'private_numeric_matrix_omitted':len(value)}
        # Even a short preview exposes case labels/predictions. Numeric arrays
        # with explicit case roles are private; fold/seed KS/AUC lists and
        # non-numeric configuration names remain useful aggregate metadata.
        if value and key in DIRECT_CASE_ARRAY_KEYS and all(v is None or isinstance(v,(int,float,bool))for v in value):
            return {'private_case_array_omitted':len(value)}
        if len(value)>20 and key in CASE_ARRAY_KEYS:
            return {'private_case_array_omitted':len(value)}
        return [clean_json(v,key)for v in value]
    if isinstance(value,str):
        return sanitize(value)
    return value


def public_content(path, logical_name=None):
    name=(logical_name or path.name).lower()
    if name.endswith('session-index.tsv') or name=='codex-a100.md' or name.startswith('sysinfo') or 'shell_history' in name or 'term_history' in name:
        return None,'private_operational_transcript'
    compressed=name.endswith('.gz')
    logical=Path(name[:-3] if compressed else name)
    raw=gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
    if logical.suffix in {'.png','.jpg','.jpeg','.webp'}:
        return raw,'figure'
    try:
        text=raw.decode('utf-8')
    except UnicodeDecodeError:
        return None,'non_utf8_private'
    changed=False
    if logical.suffix in {'.csv','.tsv'}:
        try:
            header=next(csv.reader(io.StringIO(text),delimiter='\t' if logical.suffix=='.tsv' else ','))
            columns={s.strip().lower()for s in header}
            if CASE_KEYS & columns or CASE_ID_KEYS & columns and CASE_VALUE_KEYS & columns:
                return None,'case_level_table_private'
        except (StopIteration,csv.Error):
            pass
    if logical.suffix in {'.json','.ipynb'} or logical.suffix in {'.log','.txt'} and text.lstrip().startswith(('{','[')):
        try:
            obj=json.loads(text)
            if isinstance(obj,dict) and ('tree_info' in obj or 'gradient_booster' in obj or 'learner' in obj and isinstance(obj['learner'],dict)):
                return None,'serialized_fitted_model_private'
            # Notebook outputs can contain raw DataFrame rows, embedded images,
            # or error text. Keep the source cells; retain outputs only in the
            # private original whose hash is recorded in the inventory.
            if logical.suffix=='.ipynb' and isinstance(obj,dict):
                for cell in obj.get('cells',[]):
                    if not isinstance(cell,dict):continue
                    if cell.get('outputs'):cell['outputs']=[];changed=True
                    if cell.get('execution_count')is not None:cell['execution_count']=None;changed=True
                    if 'attachments'in cell:cell.pop('attachments');changed=True
            cleaned=clean_json(obj)
            if cleaned != obj:
                changed=True
            if changed:text=json.dumps(cleaned,ensure_ascii=False,indent=2)+'\n'
        except (json.JSONDecodeError,RecursionError):
            pass
    elif logical.suffix=='.jsonl':
        lines=[]
        for line in text.splitlines():
            try:
                original=json.loads(line);cleaned=clean_json(original)
                lines.append(json.dumps(cleaned,ensure_ascii=False)if original!=cleaned else line)
                changed=changed or original!=cleaned
            except (json.JSONDecodeError,RecursionError):lines.append(line)
        if changed:text='\n'.join(lines)+'\n'
    cleaned=sanitize(text)
    changed=changed or cleaned!=text
    result=cleaned.encode()
    if compressed:
        result=gzip.compress(result,mtime=0)
    return result,'sanitized'if changed else 'unchanged'


class BundleWriter:
    """Deterministic bounded shards keep hundreds of thousands of trial records usable."""
    def __init__(self, repo, history):
        self.repo=repo;self.root=history/'bundles';self.root.mkdir(parents=True,exist_ok=True)
        self.serial=0;self.size=0;self.tar=None;self.records=[]
    def close(self):
        if self.tar is None:return
        self.tar.close();self.gz.close();self.stream.close()
        self.records.append({'public_archive':str(self.path.relative_to(self.repo)),
                             'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                             'bytes':self.path.stat().st_size,'members':self.members})
        assert self.path.stat().st_size<40*1024**2,'Unexpected oversized compressed shard'
        self.tar=None
    def add(self,member,content):
        if self.tar is None or self.size+len(content)>32*1024**2 or self.members>=10000:
            self.close();self.serial+=1;self.size=0;self.members=0
            self.path=self.root/f'trial-records-{self.serial:04d}.tar.gz'
            self.stream=self.path.open('wb');self.gz=gzip.GzipFile(filename='',mode='wb',fileobj=self.stream,mtime=0)
            self.tar=tarfile.open(fileobj=self.gz,mode='w|',format=tarfile.PAX_FORMAT)
        info=tarfile.TarInfo(member);info.size=len(content);info.mtime=0;info.mode=0o644
        self.tar.addfile(info,io.BytesIO(content));self.size+=len(content);self.members+=1
        return {'public_archive':str(self.path.relative_to(self.repo)),'archive_member':member}


class InventoryWriter:
    def __init__(self,history):
        self.history=history;self.parts=[];self.stream=None;self.rows=0;self.serial=0
    def add(self,row):
        if self.stream is None or self.rows>=50000:
            self.close();self.serial+=1;self.rows=0
            self.path=self.history/f'SOURCE_INVENTORY-{self.serial:04d}.jsonl.gz'
            self.raw=self.path.open('wb');self.stream=gzip.GzipFile(filename='',mode='wb',fileobj=self.raw,mtime=0)
        self.stream.write((json.dumps(clean_json(row),ensure_ascii=False)+'\n').encode());self.rows+=1
    def close(self):
        if self.stream is None:return
        self.stream.close();self.raw.close()
        self.parts.append({'file':self.path.name,'rows':self.rows,'bytes':self.path.stat().st_size,
                           'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()})
        self.stream=None


def export(repo, private, sources):
    history=repo/'insight/faris/history-20261028';history.mkdir(parents=True,exist_ok=True)
    assert not (history/'sources').exists() and not (history/'bundles').exists(),'Move previous generated output to private staging before a clean export'
    existing={};raw_sources={};processed={};seen_sources=set();stats=collections.Counter();total=0;public_bytes=0
    for source,inventory,fetched in sources:
        for line in inventory.open():
            item=json.loads(line)
            if item.get('sha256') and item.get('category','').startswith('public_candidate'):
                p=Path(item['path'])if source in {'local','antigravity'}else fetched/item['relative']
                if p.exists():raw_sources.setdefault(item['sha256'],p)
    writer=InventoryWriter(history);bundles=BundleWriter(repo,history)
    for source,inventory,fetched in sources:
        for line in inventory.open():
            item=json.loads(line);row={'source':source,**item}
            if item['kind']!='file':writer.add(row);total+=1;continue
            key=(source,item['path'])
            if key in seen_sources:continue
            seen_sources.add(key);total+=1
            if not item['category'].startswith('public_candidate') or source=='local' and Path(item['relative']).parts[:2]==('a100','profiles'):
                row['publication']='private_reference';stats['private_reference']+=1;writer.add(row);continue
            logical=Path(item['path']).name.lower();suffix=logical[:-3]if logical.endswith('.gz')else logical
            mode=Path(suffix).suffix
            if logical=='codex-a100.md' or logical.startswith('sysinfo') or 'shell_history'in logical or 'term_history'in logical:mode='private_operational'
            process_key=(item['sha256'],mode)
            if process_key in processed:
                saved=processed[process_key];row.update(saved)
                if saved['publication']=='archived':row['publication']='deduplicated_alias'
                stats[row['publication']]+=1;writer.add(row);continue
            p=raw_sources.get(item['sha256'],Path('/__missing_history_source__'))
            if not p.exists():
                row['publication']='missing_fetch';stats['missing_fetch']+=1;writer.add(row);continue
            if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                row['publication']='source_changed_after_inventory';stats[row['publication']]+=1;writer.add(row);continue
            content,policy=public_content(p,logical)
            row['transformation']=policy
            if content is None:
                row['publication']='private_reference';processed[process_key]={'publication':'private_reference','transformation':policy}
                stats['private_reference']+=1;writer.add(row);continue
            if len(content)>32*1024**2:
                content=gzip.compress(content,mtime=0);row['transformation']+=';compressed_for_git'
            assert len(content)<40*1024**2,(str(p),'Public blob requires explicit partitioning')
            sha=hashlib.sha256(content).hexdigest()
            if sha in existing:
                row.update(existing[sha]);row['publication']='deduplicated_alias';stats['deduplicated_alias']+=1
            else:
                relative=Path(item['relative']);assert not relative.is_absolute() and '..'not in relative.parts,relative
                bulk=(source.startswith('node') or source=='agung') and mode not in {'.py','.md','.sh','.service','.toml','.yaml','.yml','.patch','.diff'} and len(relative.parts)>=4
                if bulk:
                    row.update(bundles.add(str(Path(source)/relative),content))
                else:
                    destination=history/'sources'/source/relative
                    if ';compressed_for_git'in row['transformation']:destination=Path(str(destination)+'.gz')
                    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
                    row['public_path']=str(destination.relative_to(repo))
                row.update(publication='archived',public_sha256=sha,public_bytes=len(content));public_bytes+=len(content);stats['archived']+=1
                existing[sha]={k:v for k,v in row.items()if k in {'publication','public_path','public_archive','archive_member','public_sha256'}}
            processed[process_key]={k:v for k,v in row.items()if k in {'publication','public_path','public_archive','archive_member','public_sha256','transformation'}}
            writer.add(row)
        print(json.dumps({'stage':'export_source_complete','source':source,'records':total,'counts':dict(stats)}),flush=True)
    bundles.close();writer.close()
    index={'format':'gzip_jsonl_shards','parts':writer.parts,'rows':total}
    (history/'SOURCE_INVENTORY_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
    (history/'BUNDLE_INDEX.json').write_text(json.dumps({'bundles':bundles.records},indent=2)+'\n')
    result={'status':'exported_pending_verification','sources':[s[0]for s in sources],'counts':dict(stats),'source_records':total,
            'source_inventory':'insight/faris/history-20261028/SOURCE_INVENTORY_INDEX.json','unique_public_files':stats['archived'],
            'public_bytes':public_bytes,'bundle_count':len(bundles.records),
            'privacy_policy':'No credentials, raw datasets, case-level labels/OOF, environments or fitted weights. Originals retained privately. Content transformations recorded per source.',
            'experiment_count_warning':'Files, source references, repeated records, aliases and completed fits are distinct quantities; these counts are not independent experiments.'}
    (history/'IMPORT_STATUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True)
    parser.add_argument('--private',required=True)
    parser.add_argument('--source',action='append',nargs=3,metavar=('NAME','INVENTORY','FETCHED_ROOT'))
    args=parser.parse_args()
    export(Path(args.repo),Path(args.private),[(name,Path(inv),Path(fetched))for name,inv,fetched in args.source])
