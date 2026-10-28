"""Complete experiment/method/formula ledger, including negative and archived work.

The JSON preserves all discovered source records; Markdown embeds current
methods plus all feature catalogs and historical method documents. Missing
historical definitions remain explicitly unavailable rather than invented.
"""
import os,sys,json,csv,io,time,argparse,hashlib,ast,importlib.util,math
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7';OUT=V/'reports'
SKIP={'.git','.venv','venv','node_modules','__pycache__','data','models','checkpoints','submissions','site-packages','.cache','cache'}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 temp.write_text(value,encoding='utf-8');temp.replace(path)

def read_json(path):
 try:return json.loads(path.read_text(encoding='utf-8-sig'))
 except Exception as e:return {'source_read_error':str(e)}

def discover_archive(parent):
 records={};catalogs={};documents={};errors=[]
 for folder,dirs,files in os.walk(parent):
  dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('env') and (Path(folder)/d).resolve()!=ROOT]
  for filename in files:
   p=Path(folder)/filename;name=filename.upper();rel=p.relative_to(parent).as_posix()
   try:
    if p.stat().st_size>20_000_000:continue
    if name.endswith('.JSON') and (name=='RESULT.JSON' or name.startswith('RESULT_') or name.endswith('_PIPELINE.JSON') or any(t in name for t in ['ERROR','FAIL','ATTEMPT']) or name in ['PIPELINE.JSON','LEADERBOARD_OBSERVATIONS.JSON']):
     records[rel]=dict(source=rel,sha256=digest(p),kind='archived_json',record=read_json(p))
    elif name.endswith('.CSV') and ('EXPERIMENT' in name or ('FEATURE' in name and 'CATALOG' in name)):
     rows=list(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8-sig'))))
     item=dict(source=rel,sha256=digest(p),rows=rows)
     if 'FEATURE' in name and 'CATALOG' in name:catalogs[rel]=item
     else:records[rel]=dict(item,kind='archived_csv')
    elif name.endswith('.MD') and any(t in name for t in ['PLAN','INSIGHT','REPORT','BATCH_RESULTS','OVERFIT','EXPERIMENT_INDEX']):
     documents[rel]=dict(source=rel,sha256=digest(p),text=p.read_text(encoding='utf-8-sig'))
   except Exception as e:errors.append(dict(source=rel,error=str(e)))
 return dict(records=records,catalogs=catalogs,documents=documents,errors=errors,
  scope='Available workspace mirrors only; unavailable external/node histories cannot be asserted complete')

def plan():
 # Avoid importing the ML stack just to read literal experiment specifications.
 code=ast.parse((V/'scripts/experiments.py').read_text());rows=[]
 for node in code.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PLAN' for t in node.targets):
   for item in node.value.elts:
    rows.append({kw.arg:ast.literal_eval(kw.value) for kw in item.keywords})
 for p in sorted(V.glob('*_plan.json')):
  batch=p.stem[:-5]
  for spec in read_json(p):rows.append(dict(spec,batch=batch,feature_catalog=batch+'/reports/FEATURE_CATALOG.csv'))
 assert len({s['id'] for s in rows})==len(rows)
 return sorted(rows,key=lambda s:int(s['id'][1:]))

def esc(value):return str(value).replace('|','/').replace('\n',' ')

def metric(record,*keys):
 for container in [record,record.get('OOF',{}),record.get('metrics',{}),record.get('evaluation',{})]:
  if isinstance(container,dict):
   for key in keys:
    value=container.get(key)
    if value is not None and value!='':return value
 return None

def generate():
 OUT.mkdir(exist_ok=True,parents=True);seed=V/'MASTER_ARCHIVE_SNAPSHOT.json'
 archives=read_json(seed) if seed.exists() else dict(records={},catalogs={},documents={},errors=[])
 current_archive=discover_archive(ROOT.parent)
 for kind in ['records','catalogs','documents']:archives.setdefault(kind,{}).update(current_archive[kind])
 archives.setdefault('errors',[]).extend(current_archive['errors'])
 policy=read_json(V/'FEATURE_POLICY.json')
 public_path=ROOT/'reports/LEADERBOARD_OBSERVATIONS.json'
 public_observations=read_json(public_path) if public_path.exists() else []
 if not isinstance(public_observations,list):raise ValueError('Leaderboard observations must remain a list')
 all_catalogs={};docs={}
 for p in V.rglob('FEATURE_CATALOG.csv'):
  all_catalogs[p.relative_to(ROOT).as_posix()]=list(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8-sig'))))
 for p in V.glob('*PLAN.md'):docs[p.relative_to(ROOT).as_posix()]=p.read_text(encoding='utf-8')
 for p in ROOT.glob('*.md'):
  if p.name in ['README.md','WHY_BASELINE_WORKS.md','BASELINE_LOCK.md']:docs[p.name]=p.read_text(encoding='utf-8')
 policy_module=V/'scripts/feature_policy.py';spec=importlib.util.spec_from_file_location('ledger_policy',policy_module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 active=[]
 for definition in plan():
  exp=definition['id'];folder=V/'runs'/exp;result=folder/'RESULT.json';status=folder/'STATUS.json'
  r=module.effective_result(read_json(result)) if result.exists() else None
  state=read_json(status) if status.exists() else {'status':'QUEUED'}
  process=OUT/f'PROCESS_{exp}.json'
  if process.exists():
   p=read_json(process)
   if p.get('status')=='FAILED' and r is None:state=p
  source=V/definition.get('feature_catalog','reports/FEATURE_CATALOG.csv')
  catalog=all_catalogs.get(source.relative_to(ROOT).as_posix(),[])
  features=[x for x in catalog if x.get('family') in definition.get('families',[])]
  if 'input_features' in definition:
   features=[x for x in catalog if x.get('feature_name') in definition['input_features']]
  if 'only' in definition:features=[x for x in catalog if x.get('feature_name') in definition['only']]
  if r is not None:features=[x for x in catalog if x.get('feature_name') in r['new_features']]
  record=dict(id=exp,definition=definition,status='COMPLETE' if r else state.get('status','QUEUED'),
   method_formula='p_final=.75*sigmoid(logit(p_frozen_Kumo)+strength*X_correction@beta)+.25*p_original_spline77; test mean5fold',
   objective='mean(logaddexp(0,offset+Xbeta)-y*(offset+Xbeta)) + penalty/2*sum(beta[1:]**2); fit full correction, shrink at inference',
   validation='Identical5fold seed31027;50k context,6k correction,14k validation; frozen backbone90 inputs; additions affect correction branch',
   feature_definitions=features,feature_definition_status='CATALOG_AVAILABLE' if catalog else 'BATCH_PLAN_AND_SOURCE_AVAILABLE_CATALOG_NOT_READY',
   feature_catalog=source.relative_to(ROOT).as_posix(),result=r,state=state,source_result_sha256=digest(result) if result.exists() else None,
   source_code='v7/scripts/'+definition.get('batch','features')+('_features.py' if definition.get('batch') else '.py'),
   matched_coverage_control=read_json(folder/'MATCHED_CONTROL.json') if (folder/'MATCHED_CONTROL.json').exists() else None)
  if r and r.get('modeling_formula'):
   record['method_formula']=r['modeling_formula'];record['objective']=r.get('modeling_objective',record['objective'])
   record['source_code']=r.get('controller_source',record['source_code'])
   record['error_analysis']=r.get('error_analysis')
   if r.get('in_context_refitted'):
    record['validation']='Identical5fold seed31027;50k context,6k correction,14k validation; fresh frozen Kumo in-context fitting, context columns recorded in result; original spline unchanged'
  elif definition.get('formula'):
   record['method_formula']=definition['formula'];record['source_code']=definition.get('source_script',record['source_code'])
  record['public_leaderboard_observations']=[p for p in public_observations if r and p.get('submission_sha256')==r.get('submission_sha256')]
  active.append(record)
 queues={p.name:read_json(p) for p in OUT.glob('*QUEUE.json')};feature_stages={p.relative_to(ROOT).as_posix():read_json(p) for p in V.glob('*/reports/FEATURE_STATUS.json')}
 incidents={p.name:read_json(p) for p in OUT.glob('*.json') if any(t in p.name.upper() for t in ['FAILURE','ERROR'])}
 value=dict(updated_UTC=datetime.now(timezone.utc).isoformat(),scope='All active planned/completed/failed variants and available historical mirrors; no score-based filtering',
  DPD_policy=policy,locked_baseline=read_json(ROOT/'BASELINE_LOCK.json'),active_experiments=active,all_active_feature_catalogs=all_catalogs,
  active_method_documents=docs,archive=archives,queue_states=queues,feature_build_stages=feature_stages,incident_records=incidents,
  public_leaderboard_observations=public_observations,
  best_reported_public=max((p for p in public_observations if isinstance(p.get('public_score'),(int,float))),key=lambda p:(p['public_score'],bool(p.get('downloaded_csv_sha_verified')),bool(p.get('independently_verified_competition_receipt')),p.get('recorded_UTC',p.get('UTC',''))),default=None),
  submission_recommendations={p.name:read_json(p) for p in OUT.glob('SUBMISSION_RECOMMENDATION_*.json')},
  caveats=['Archive entries may use different validation folds/models; no global cross-project leaderboard','Missing historical formulas/results are explicitly unavailable','Completed result source records preserved; policy view does not alter predictions','JSON result files, CSV rows and fold records can describe the same experiment; record counts are not unique experiment counts'])
 def clean(x):
  if isinstance(x,float) and not math.isfinite(x):return None
  if isinstance(x,dict):return {k:('[REDACTED]' if k.lower() in ['password','api_key','access_token','refresh_token'] else clean(v)) for k,v in x.items()}
  if isinstance(x,list):return [clean(v) for v in x]
  return x
 value=clean(value)
 atomic(OUT/'EXPERIMENT_MASTER_LOG.json',json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False))
 done=sum(x['status']=='COMPLETE' for x in active);failed=sum(x['status']=='FAILED' for x in active)
 lines=['# Catatan induk seluruh eksperimen','',f"Diperbarui UTC: {value['updated_UTC']}",'',
  f"Aktif: {len(active)} rencana, {done} selesai, {failed} gagal. Semua hasil naik/turun/tidak berubah tetap dicatat. Hasil yang belum tersedia tidak diisi nol.",'',
  f"Arsip: {len(archives['records'])} berkas catatan, {len(archives['catalogs'])} katalog fitur, {len(archives['documents'])} dokumen metode. Ini jumlah sumber, bukan jumlah eksperimen unik.",'',
  'Validasi antar-proyek tidak selalu sama. Skor public hanya berlaku untuk file/hash yang dilaporkan. Tidak ada label test tersembunyi. DPD diizinkan tanpa karantina; ketidakpastian snapshot tetap tercatat.','',
  '## Ringkasan eksperimen aktif','', '| ID | Status | Metode | KS | Delta KS | AUC | Delta AUC |','|---|---|---|---:|---:|---:|---:|']
 flat=[]
 for x in active:
  r=x['result'];m=r['OOF'] if r else {};d=x['definition'];row=dict(scope='ACTIVE_MATCHED_CV',experiment=x['id'],status=x['status'],method=d['label'].replace('QUARANTINED ',''),
   KS=m.get('KS'),delta_KS=r['delta_KS'] if r else None,AUC=m.get('AUC'),delta_AUC=r['delta_AUC'] if r else None,
   penalty=d['penalty'],strength=d['strength'],method_formula=x['method_formula'],feature_formulas=json.dumps(x['feature_definitions'],ensure_ascii=True),
   source=f"v7/runs/{x['id']}/RESULT.json",submission=r['submission'] if r else None,
   reported_public_KS=x['public_leaderboard_observations'][-1]['public_score'] if x['public_leaderboard_observations'] else None,
   public_score_source=x['public_leaderboard_observations'][-1].get('source') if x['public_leaderboard_observations'] else None)
  flat.append(row);lines.append('| '+' | '.join(esc(row[k]) if row[k] is not None else 'BELUM TERSEDIA' for k in ['experiment','status','method','KS','delta_KS','AUC','delta_AUC'])+' |')
 lines+=['','## Leaderboard public yang dilaporkan pengguna','',
  'Skor berikut adalah laporan pengguna untuk hash CSV tertentu, terpisah dari OOF. Memilih kandidat setelah melihat leaderboard tidak menghasilkan bukti validasi independen.','',
  '| Variant | Public KS | Submission SHA256 |','|---|---:|---|']
 for p in public_observations:lines.append('| '+esc(p.get('variant',''))+' | '+esc(p.get('public_score'))+' | '+esc(p.get('submission_sha256',''))+' |')
 lines+=['','## Metode dan hasil per eksperimen aktif','']
 for x in active:
  d=x['definition'];r=x['result'];lines += [f"### {x['id']} — {d['label'].replace('QUARANTINED ','')}",'',f"Status: {x['status']}. Keluarga: {d['families']}. L2={d['penalty']}, strength={d['strength']}.",'',f"Rumus model: `{x['method_formula']}`",'',f"Objective: `{x['objective']}`",'',x['validation'],'']
  if r:
   lines += [f"Hasil OOF: `{json.dumps(r['OOF'])}`",'',f"Delta KS={r['delta_KS']:+.8f}; delta AUC={r['delta_AUC']:+.8f}; menang KS {r['fold_wins']}/5 fold.",'',
    f"Confidence interval: `{json.dumps(r['confidence_interval'])}`",'',f"Overfitting: `{json.dumps(r['overfitting_assessment'])}`",'',
    f"Submission: `{r['submission']}`; SHA256 `{r['submission_sha256']}`. Checkpoint: `v7/runs/{x['id']}/foldN_correction.joblib`; OOF: `v7/runs/{x['id']}/OOF.csv`.",'',
    '| Fold | KS | Delta KS | AUC | Delta AUC |','|---|---:|---:|---:|---:|']
   for f in r['folds']:lines.append(f"| {f['fold']} | {f['metrics']['KS']:.8f} | {f['delta_KS']:+.8f} | {f['metrics']['AUC']:.8f} | {f['delta_AUC']:+.8f} |")
   lines+=['']
  else:lines+=['KS/AUC/CI: BELUM TERSEDIA. Status mentah: `'+json.dumps(x['state'])+'`.','']
  if x['public_leaderboard_observations']:lines+=['Laporan public untuk CSV yang sama: `'+json.dumps(x['public_leaderboard_observations'])+'`.','']
  if x['matched_coverage_control']:lines+=['Perbandingan terhadap coverage-control: `'+json.dumps(x['matched_coverage_control'])+'`.','']
  lines+=['| Fitur | Rumus | Cutoff / batasan |','|---|---|---|']
  for f in x['feature_definitions']:lines.append('| '+esc(f.get('feature_name',''))+' | '+esc(f.get('formula','TIDAK TERCATAT'))+' | '+esc(f.get('cutoff_rule',''))+' '+esc(f.get('caveat',''))+' |')
  if not x['feature_definitions']:lines+=['| Belum ada katalog / kontrol | Lihat dokumen batch lengkap dan sumber kode di bawah | Tidak mengarang definisi yang belum dicatat |']
  lines+=['']
 lines+=['## Seluruh katalog fitur, termasuk tidak dipakai atau tidak menaikkan skor','']
 for source,rows in {**all_catalogs,**{k:v['rows'] for k,v in archives['catalogs'].items()}}.items():
  lines += ['### '+source,'','| Fitur | Rumus | Status / sumber / cutoff |','|---|---|---|']
  for f in rows:
   name=f.get('feature_name',f.get('feature',f.get('name','TIDAK TERCATAT')));formula=f.get('formula',f.get('definition','TIDAK TERCATAT DI SUMBER'))
   lines.append('| '+esc(name)+' | '+esc(formula)+' | '+esc({k:v for k,v in f.items() if k not in ['feature_name','feature','name','formula','definition']})+' |')
  lines+=['']
 lines+=['## Catatan hasil historis — validasi terpisah','']
 for source,record in archives['records'].items():
  lines+=['### '+source,'',f"SHA256: `{record['sha256']}`. Status efektif: ARSIP; last state/source outcome dipertahankan.",'']
  records=record.get('rows',[record.get('record',{})])
  for index,r in enumerate(records):
   if not isinstance(r,dict):continue
   exp=r.get('experiment_id',r.get('variant',r.get('name',str(index))));ks=metric(r,'OOF_KS','KS','ks');auc=metric(r,'OOF_AUC','AUC','auc')
   flat.append(dict(scope='ARCHIVED_RECORD',experiment=exp,status=r.get('status',r.get('stage','AS_RECORDED')),method=r.get('method',r.get('name',r.get('variant','TIDAK TERCATAT'))),KS=ks,AUC=auc,source=source))
  # Preserve every original field, including per-fold negative outcomes and warnings.
  lines+=['```json',json.dumps(records,indent=2,ensure_ascii=True),'```','']
 lines+=['## Insiden dan perbaikan — riwayat kegagalan tidak dihapus','', '```json',json.dumps(incidents,indent=2,ensure_ascii=True),'```','']
 lines+=['## Dokumen metode dan formula seluruh batch','']
 for source,text in {**docs,**{k:v['text'] for k,v in archives['documents'].items()}}.items():lines+=['### Sumber: '+source,'',text,'']
 lines+=['## Kelengkapan dan batas pencatatan','',*['- '+s for s in value['caveats']],'',
  'Definisi yang tidak tersedia di arsip ditandai TIDAK TERCATAT; tidak dibuat klaim formula baru. Semua source JSON/CSV, dokumen dan katalog juga dipertahankan lengkap dalam EXPERIMENT_MASTER_LOG.json.','']
 atomic(OUT/'EXPERIMENT_MASTER_LOG.md','\n'.join(lines));columns=list(dict.fromkeys(k for r in flat for k in r))
 text=io.StringIO();writer=csv.DictWriter(text,fieldnames=columns);writer.writeheader();writer.writerows(flat);atomic(OUT/'EXPERIMENT_MASTER_LOG.csv',text.getvalue())
 summary=dict(status='READY',pid=os.getpid(),updated_UTC=value['updated_UTC'],active_planned=len(active),active_complete=done,active_failed=failed,
  archive_record_files=len(archives['records']),archive_feature_catalogs=len(archives['catalogs']),archive_method_documents=len(archives['documents']),all_active_catalog_features=sum(map(len,all_catalogs.values())),no_score_based_filter=True)
 modeling=[]
 for x in active:
  r=x['result']
  if not r or int(x['id'][1:])<120:continue
  interval=r['confidence_interval']['delta_KS']
  modeling.append(dict(experiment=x['id'],method=x['definition']['label'],KS=r['OOF']['KS'],AUC=r['OOF']['AUC'],delta_KS=r['delta_KS'],delta_AUC=r['delta_AUC'],KS_fold_wins=r['fold_wins'],head_gap=sum(f['train_validation_head_KS_gap'] for f in r['folds'])/len(r['folds']),CI_low=interval[0],CI_high=interval[1],submission=r['submission']))
 if modeling:
  text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(modeling[0]));writer.writeheader();writer.writerows(modeling);atomic(V/'modeling/reports/MODELING_EXPERIMENTS.csv',text.getvalue())
 for queue_file in [V/'KUMO_QUEUE_10.json',V/'KUMO_QUEUE_15.json']:
  if not queue_file.exists():continue
  q=read_json(queue_file);states={x['id']:x for x in active}
  for row in q['experiments']:
   item=states[row['id']];r=item['result'];row.update(status=item['status'],KS=r['OOF']['KS'] if r else None,AUC=r['OOF']['AUC'] if r else None)
  q['completed']=sum(x['status']=='COMPLETE' for x in q['experiments']);q['updated_UTC']=value['updated_UTC']
  q['cache_only_experiments']=['V172','V177'];q['cache_only_required_RAM_GiB']=.6;q['cache_only_required_VRAM_GiB']=0.
  atomic(queue_file,json.dumps(q,indent=2))
  total=q['total_experiments'];lines=[f'# Antrean{total} eksperimen Kumo100% frozen','',f"Selesai{q['completed']}/{total}. BaselineV25; target pooledKS>=0.531. Auto-start saat resource cukup. Cache-only V172/V177 dan adapterV179–V183 tidak memerlukan GPU.",'','| ID | Metode | Status | KS | AUC |','|---|---|---|---:|---:|']
  for row in q['experiments']:lines.append('| '+' | '.join(esc(row[k]) if row[k] is not None else 'BELUM TERSEDIA' for k in ['id','method','status','KS','AUC'])+' |')
  atomic(queue_file.with_suffix('.md'),'\n'.join(lines))
 atomic(OUT/'MASTER_LOG_STATUS.json',json.dumps(summary,indent=2));return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');p.add_argument('--watch',action='store_true');args=p.parse_args()
 if args.snapshot:
  value=discover_archive(ROOT.parent);atomic(V/'MASTER_ARCHIVE_SNAPSHOT.json',json.dumps(value,ensure_ascii=True));print({k:len(value[k]) for k in ['records','catalogs','documents']})
 else:
  while True:
   try:print(json.dumps(generate()),flush=True)
   except Exception as e:
    atomic(OUT/'MASTER_LOG_STATUS.json',json.dumps(dict(status='RETRY',pid=os.getpid(),error=str(e)),indent=2))
    if not args.watch:raise
   if not args.watch:break
   time.sleep(30)
