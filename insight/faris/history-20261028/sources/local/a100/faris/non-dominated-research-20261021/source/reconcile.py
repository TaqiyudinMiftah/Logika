"""Reconcile retained experiment evidence without treating copies as new trials."""
from pathlib import Path
import json,gzip,hashlib,time,collections,re,csv
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'inventory'
def stable(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(stable(x).encode()).hexdigest()
def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def normalize_status(value):
    value=str(value).lower()
    if value in ['complete','completed','finished','passed']:return 'complete'
    if value in ['failed','error','invalid']:return 'failed'
    if value in ['partial','interrupted','stopped','pruned','killed','timeout']:return 'partial'
    if value in ['planned','pending','queued','waiting','unstarted','not_started','budget_skipped']:return 'planned'
    if value in ['unschedulable','excluded']:return 'excluded'
    if value in ['running','active','training']:return 'reported_running'
    return 'reported_unverified'
def clean(value):
    """Parameter identity, not yet a qualified executed-behavior descriptor."""
    if isinstance(value,dict):
        ignore={'id','seed','random_state','random_seed','name','label','hypothesis','control','is_control',
                'changes','source','source_sha256','directory','host','node_id','started','finished','attempt',
                'trial_number','protocol','resources','device','execution_timeout_seconds','until'}
        return {k:clean(v) for k,v in sorted(value.items()) if k not in ignore}
    if isinstance(value,list):return [clean(v) for v in value]
    return value

def main():
    OUT.mkdir(exist_ok=True);sources=[];objects={};skipped=[]
    for mf in sorted((ROOT/'snapshots').glob('*/manifest.json')):
        m=read(mf)
        for s in m['sources']:
            path=ROOT/s['snapshot']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==s['snapshot_sha']
            try:obj=read(path)
            except (ValueError,UnicodeError) as e:skipped.append(dict(source=s['source'],reason=str(e)));continue
            key=s['host']+'::'+s['source'];sources.append(s);objects[key]=(s,obj)
    records={};observations=0;conflicts=[]
    def add(key,kind,status,cohort,configuration,source,row=None,metrics=None,prediction=None,parents=None):
        nonlocal observations
        observations+=1
        ref=dict(host=source['host'],source=source['source'],source_sha=source['source_sha'],row=row)
        if key in records:
            old=records[key];old['references'].append(ref)
            if metrics:
                if old.get('reported_metrics') and any(abs(old['reported_metrics'][k]-metrics[k])>1e-10
                    for k in set(old['reported_metrics'])&set(metrics) if isinstance(metrics[k],(int,float))):
                    conflicts.append(dict(id=key,type='metric_disagreement',source=ref))
                old['reported_metrics']=metrics
            if prediction:old['prediction_ref']=prediction
            if old['configuration'] is None and configuration is not None:old['configuration']=configuration
            # Authoritative task state wins over a scoreboard-only completion.
            if source['category'] in ['legacy_ledger','campaign_state']:old['status']=status
            return old
        result=dict(id=key,kind=kind,status=status,cohort=cohort,configuration=configuration,
            references=[ref],reported_metrics=metrics,prediction_ref=prediction,parents=parents or [],
            fresh_score_verified=False,rank_eligible=False,
            identity_scope='Retained task/recipe identity. Repeated folds and seeds are not distinct parameter designs.')
        records[key]=result;return result
    def metrics(row):
        if not isinstance(row,dict):return None
        out={}
        for target,names in [('KS',['KS','ks','ks_statistic','score']),('AUC',['AUC','roc_auc','auc']),('logloss',['logloss','log_loss'])]:
            for name in names:
                v=row.get(name)
                if isinstance(v,(int,float)):out[target]=float(v);break
        return out or None
    # Ledgers own lifecycle states; scoreboards add measurements, not new attempts.
    for s,obj in objects.values():
        if s['category']!='legacy_ledger':continue
        campaign=Path(s['source']).parent.name
        for row in obj:
            task=row['task'];mode=row['mode'];recipe=task.get('recipe')
            cohort='legacy56k_threefold' if mode in ['cv','nested','reference'] else 'legacy56k_'+mode
            kind='nested_ensemble' if mode=='nested' else 'pipeline' if mode=='cv' else 'assessment_'+mode
            add('meta/'+campaign+'/'+mode+'/'+row['id'],kind,normalize_status(row['state']),cohort,
                recipe,s,row=row['id'],prediction=str(Path(s['source']).parent/'runs'/row['id']/'oof.npz'))
    for s,obj in objects.values():
        if s['category']!='legacy_report' or not Path(s['source']).name.startswith('scoreboard_') or not isinstance(obj,list):continue
        campaign=Path(s['source']).parents[1].name
        for row in obj:
            if not isinstance(row,dict) or not row.get('id'):continue
            kind=row.get('kind',Path(s['source']).stem.replace('scoreboard_',''))
            mode={'pipeline':'cv','nested_ensemble':'nested','reference':'reference'}.get(kind,kind)
            cohort='legacy56k_threefold' if kind in ['pipeline','fixed_ensemble','nested_ensemble','reference'] else 'legacy56k_'+kind
            recipe=row.get('recipe');p=Path(s['source']).parents[1]/'runs'/row['id']/'oof.npz'
            add('meta/'+campaign+'/'+mode+'/'+row['id'],kind,normalize_status(row.get('state','complete')),cohort,
                recipe,s,row=row['id'],metrics=metrics(row.get('pooled_oof',row.get('metrics',row))),prediction=str(p),
                parents=(recipe or {}).get('parents',[]))
    # Every ordinary campaign job is retained once per campaign instance. A repeat
    # and a temporal check remain assessments, not new training designs.
    jobs_by_external_id={}
    for s,obj in objects.values():
        if s['category']!='campaign_state' or not isinstance(obj,dict) or not isinstance(obj.get('jobs'),dict):continue
        root=Path(s['source']).parent;instance=obj.get('id',root.parent.name)
        phase=root.name
        for jid,job in obj['jobs'].items():
            protocol=job.get('protocol','unknown');family=job.get('family','unknown')
            is_horizontal='horizontal' in instance
            cohort=('original70k_fivefold' if is_horizontal else
                'legacy56k_threefold' if protocol=='screen' else 'legacy56k_'+protocol)
            key=f'campaign/{instance}/{phase}/{jid}'
            configuration=dict(family=family,params=job.get('params',{}))
            row=add(key,'fold_fit' if is_horizontal else 'nested_ensemble' if family=='nested' else 'pipeline' if protocol=='screen' else 'assessment_'+protocol,
                normalize_status(job.get('status')),cohort,configuration,s,row=jid,
                metrics=metrics(dict(score=job['score'])) if isinstance(job.get('score'),(int,float)) else None,
                prediction=str(root/job['directory']/'validation.npz') if job.get('directory') else None)
            row['attempt_number']=job.get('attempt');row['seed']=job.get('seed');row['resources_reported']=dict(
                elapsed_seconds=(job.get('finished',0)-job.get('started',0)) if job.get('started') and job.get('finished') else None,
                peak_ram_mb=job.get('peak_ram_mb'),peak_vram_mb=job.get('peak_vram_mb'))
            external=s['host']+'_'+jid
            if 'exploration' in str(root):jobs_by_external_id[external]=key
    # Current campaign individual and actuarial scoreboards overlap by design.
    for s,obj in objects.values():
        name=Path(s['source']).name
        if s['category']!='study_report' or not name.startswith('scoreboard_') or not isinstance(obj,list):continue
        if 'exploration-12h-20261017' not in s['source']:continue
        if name in ['scoreboard_individual.json','scoreboard_actuarial.json']:
            for row in obj:
                if not row.get('id'):continue
                protocol=row.get('protocol','screen');kind='pipeline' if protocol=='screen' else 'assessment_'+protocol
                cohort='legacy56k_threefold' if protocol=='screen' else 'legacy56k_'+protocol
                key=jobs_by_external_id.get(row['id'],'exploration-retained/'+row['id'])
                add(key,kind,'complete',cohort,dict(family=row.get('family'),params=row.get('params')),
                    s,row=row['id'],metrics=metrics(row.get('evaluation',{})))
        elif name=='scoreboard_fixed.json':
            for row in obj:
                recipe=row.get('spec',{});p=str(Path(s['source']).parent/(row['id']+'.npy'))
                add('exploration-fixed/'+row['id'],'fixed_ensemble','complete','legacy56k_threefold',recipe,
                    s,row=row['id'],metrics=metrics(row),prediction=p,parents=recipe.get('parents',[]))
        elif name=='scoreboard_nested.json':
            for row in obj:
                add('exploration-nested/'+row['id'],'nested_ensemble','complete','legacy56k_threefold',
                    row.get('spec',row.get('params')),s,row=row['id'],metrics=metrics(row.get('evaluation',row)))
    # Teammate master log and individual RESULT files are views of the same run.
    teammate_submissions={}
    def teammate(result,source,version,identifier=None):
        identifier=identifier or result.get('experiment',result.get('id'))
        if not identifier or not isinstance(result.get('OOF'),dict):return
        file_sha=result.get('submission_sha256')
        key=teammate_submissions.get(file_sha) if file_sha else None
        key=key or 'teammate/'+version+'/'+str(identifier)
        if file_sha:teammate_submissions[file_sha]=key
        row=add(key,'pipeline','complete','original70k_fivefold',
            dict(spec=result.get('spec'),columns=result.get('columns'),method=result.get('method_formula',
                'frozen_Kumo90_offset_correction_75_25_spline77')),source,row=identifier,metrics=metrics(result['OOF']))
        row['submission_sha256']=file_sha
        row['source_declared']=dict(source_sha=result.get('source_sha256'),feature_matrix_sha=result.get('feature_matrix_sha256'))
    for s,obj in objects.values():
        if isinstance(obj,dict) and isinstance(obj.get('active_experiments'),list):
            for row in obj['active_experiments']:
                if isinstance(row.get('result'),dict):teammate(row['result'],s,'v7',row.get('id'))
                else:add('teammate/v7/'+str(row.get('id')),'pipeline',normalize_status(row.get('status')),
                    'original70k_fivefold',row.get('definition'),s,row=row.get('id'))
        if Path(s['source']).name=='RESULT.json' and '/v7/runs/' in s['source']:
            teammate(obj,s,'v7',Path(s['source']).parent.name)
    # Current targeted studies: each recipe is a design; its folds are evidence.
    targeted={'teammate-approach-20261020','ri[REDACTED_CREDENTIAL]','ri[REDACTED_CREDENTIAL]',
              'coverage-transfer-20261021','nested-calibration-20261020','measurement-risk-20261020'}
    for s,obj in objects.values():
        path=Path(s['source'])
        if path.name!='result.json' or not isinstance(obj,dict):continue
        study=path.parent.name
        if study in ['results','PC3-results']:study=path.parents[1].name
        if study=='feature-study':study='teammate-approach-20261020/feature-study'
        if study not in targeted and study!='teammate-approach-20261020/feature-study':continue
        values=obj.get('metrics',{})
        if not isinstance(values,dict):continue
        for name,mm in values.items():
            if not isinstance(mm,dict):continue
            # Reused controls are evidence references, not newly trained recipes.
            reused=name in ['baseline','historical','published_V25','published_original','signal','null','teacher']
            if study=='ri[REDACTED_CREDENTIAL]' and name in ['signal','null']:reused=False
            if study=='ri[REDACTED_CREDENTIAL]' and name=='matched_query':reused=False
            record=add('targeted/'+study+'/'+name,'reused_reference' if reused else 'pipeline',
                normalize_status(obj.get('status','complete')),'original70k_fivefold',
                dict(study=study,recipe=name),s,row=name,metrics=metrics(mm))
            record['configuration_resolution']='Requires source recipe resolution before novelty ranking; names do not manufacture novelty.'
    # Import older teammate result formats, retaining fold-level evidence as
    # fold-level evidence. A copied checkout contributes another reference.
    def canonical_repo(path):
        for prefix in ['/home/taqiyudinmiftah/adit/Logika/','/home/taqiyudinmiftah/Logika/']:
            if path.startswith(prefix):return path[len(prefix):]
        return path
    def historical_walk(value,source,locator='$'):
        if isinstance(value,dict):
            mm=metrics(value.get('metrics',value))
            # Never interpret an isolated public score or a bootstrap bound as
            # a fitted OOF score. Require both discrimination metrics.
            if mm and 'KS' in mm and 'AUC' in mm:
                fold=value.get('fold');rows=value.get('rows',value.get('metrics',{}).get('rows'))
                kind='fold_evidence' if fold is not None else 'historical_measurement'
                cohort=('original70k_fivefold_reported' if rows==70000 and 'large_latent_reproduction' in source['source']
                    else 'historical_protocol_unresolved')
                relative=canonical_repo(source['source'])
                label=value.get('variant',value.get('name',value.get('model',locator)))
                # Identity is a retained measurement, not a semantic design.
                key='historical/'+relative+'/'+locator
                config={k:value[k] for k in ['variant','model','parameters','recipe','kumo_features','spline_features',
                    'input_features','features','estimators','context_rows','sampling','feature_selection','kind'] if k in value}
                row=add(key,kind,'reported_unverified',cohort,config,source,row=locator,metrics=mm)
                row.update(reported_label=label,reported_fold=fold,reported_rows=rows,
                    point_in_time_declared=value.get('trusted_point_in_time',value.get('trusted')),
                    configuration_resolution='Historical metadata only; executed behavior and protocol require source/prediction verification.')
                # Fold results nested in aggregate reports are retained as
                # references within that report, not another full recipe.
                return
            for key,child in value.items():
                if key in ['top_model_gain_features','fold_results','folds','confidence_interval','CI','reliability']:continue
                if isinstance(child,(dict,list)):historical_walk(child,source,locator+'/'+str(key))
        elif isinstance(value,list):
            for i,child in enumerate(value):historical_walk(child,source,locator+'/'+str(i))
    already={(ref['host'],ref['source']) for row in records.values() for ref in row['references']}
    for s,obj in objects.values():
        if s['category']=='teammate_repo_history' and (s['host'],s['source']) not in already:
            if '/v7/' in s['source']:continue
            if any(t in Path(s['source']).name.lower() for t in ['result','summary','score']):historical_walk(obj,s)
    # Do not discard unknown result formats: expose all unresolved metadata sources.
    covered={(ref['host'],ref['source']) for row in records.values() for ref in row['references']}
    unresolved=[dict(host=s['host'],source=s['source'],category=s['category']) for s in sources
        if (s['host'],s['source']) not in covered and (s['category'] in ['teammate_repo_history','study_metadata']
        and any(t in Path(s['source']).name.lower() for t in ['result','summary','score','status']))]
    dispositions=[]
    for item in unresolved:
        obj=objects[item['host']+'::'+item['source']][1];name=Path(item['source']).name.lower()
        if name=='status.json' and '/v7/runs/' in item['source']:
            reason='Lifecycle sidecar for an existing V7 recipe; not another experiment.'
        elif 'status' in name:
            reason='Queue, feature-build or controller status; no independent fitted-recipe metric.'
        elif 'public' in name:
            reason='Public leaderboard observation; not private validation or a new training recipe.'
        elif '/final/' in item['source']:
            reason='Full-data refit artifact; cannot be ranked as an OOF measurement.'
        elif name=='spline_result.json':
            reason='Component fit metadata; final ensemble evidence is retained separately.'
        else:
            reason='Metadata without recognized KS plus AUC measurement; inspect before inclusion.'
        dispositions.append(dict(**item,reason=reason,keys=list(obj)[:40] if isinstance(obj,dict) else None))
    dump(OUT/'metadata_dispositions.json',dispositions)
    # Raw parameter grouping is provisional. It is explicitly distinct from the
    # executed-behavior descriptor identity calculated in the next stage.
    groups=collections.defaultdict(list)
    for key,row in records.items():
        if row['configuration'] is not None:
            identity=digest(dict(cohort=row['cohort'],kind=row['kind'],parameters=clean(row['configuration'])))
            row['parameter_identity']=identity;groups[identity].append(key)
        row['reference_count']=len(row['references'])
    bycohort={};bykind=collections.Counter();bystatus=collections.Counter()
    for row in records.values():
        group=bycohort.setdefault(row['cohort'],dict(records=0,complete=0,with_reported_score=0,kind_counts={}))
        group['records']+=1;group['complete']+=int(row['status']=='complete');group['with_reported_score']+=int(bool(row['reported_metrics']))
        group['kind_counts'][row['kind']]=group['kind_counts'].get(row['kind'],0)+1
        bykind[row['kind']]+=1;bystatus[row['status']]+=1
    archive=list(records.values());archive.sort(key=lambda x:x['id'])
    with gzip.open(OUT/'retained_records.json.gz','wt',compresslevel=5) as file:json.dump(archive,file,separators=(',',':'),allow_nan=False)
    duplicates=[dict(parameter_identity=k,record_ids=v) for k,v in groups.items() if len(v)>1]
    dump(OUT/'parameter_duplicates.json',duplicates);dump(OUT/'unresolved_sources.json',unresolved);dump(OUT/'conflicts.json',conflicts)
    summary=dict(created=time.time(),source_files=len(sources),source_reference_rows=observations,
        retained_recipe_and_assessment_records=len(records),merged_repeated_references=observations-len(records),
        by_status=dict(bystatus),by_kind=dict(bykind),by_cohort=bycohort,
        provisional_parameter_groups=len(groups),parameter_repeat_groups=len(duplicates),
        unresolved_metadata_sources=len(unresolved),conflicts=len(conflicts),parse_errors=skipped,
        inventory_complete=False,reason='Resolve remaining historical formats, prediction metrics, and executed-behavior descriptors before claiming an exhaustive distinct-experiment total.',
        new_training_allowed=False,new_training_started=False,
        scientific_note='Reported scores are not newly verified; incompatible cohorts are separate; parameter identities are not yet qualified semantic design identities.')
    dump(OUT/'reconciliation.json',summary)
    lines=['# Existing experiment reconciliation (in progress)','',f'Retained recipe/assessment records: {len(records):,}.',
        f'Source reference rows merged: {observations-len(records):,}.',
        f'Unresolved historical metadata sources: {len(unresolved):,}.','',
        '| Cohort | Records | Completed | With reported metrics |','|---|---:|---:|---:|']
    for name,c in sorted(bycohort.items()):lines.append(f'| {name} | {c["records"]} | {c["complete"]} | {c["with_reported_score"]} |')
    lines+=['','Counts include separate assessment records and reused controls; they are not a count of independent completed model designs.',
        'No new training is admitted before reconciliation, scoring checks and novelty calculations finish.',
        'Unresolved formats and verification gaps remain explicit. The .54 target has not been attained.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
