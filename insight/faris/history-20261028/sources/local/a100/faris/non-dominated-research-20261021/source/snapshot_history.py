"""Read-only history snapshot. No training, submission, or existing-state writes."""
from pathlib import Path
import argparse,json,gzip,hashlib,time,sqlite3,os

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def dump(p,value):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def snapshot(host,bases,repos=()):
    dest=ROOT/'snapshots'/host;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'manifest.json').exists(),'Preserve completed snapshot'
    manifest=[];missing=[]
    def capture(path,category,transform=None):
        path=Path(path)
        if not path.exists():missing.append(str(path));return
        raw=path.read_bytes()
        if path.suffix=='.json' and transform is not None:
            obj=json.loads(raw)
            if transform:obj=transform(obj)
            encoded=json.dumps(obj,separators=(',',':'),allow_nan=False).encode()
        else:encoded=raw
        key=hashlib.sha256(str(path).encode()).hexdigest()[:16]
        out=dest/(key+'-'+path.name+'.gz')
        with gzip.open(out,'wb',compresslevel=4) as f:f.write(encoded)
        manifest.append(dict(source=str(path),category=category,source_sha=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),snapshot=str(out.relative_to(ROOT)),snapshot_sha=sha(out),host=host))
    for base in bases:
        base=Path(base)
        if not base.exists():missing.append(str(base));continue
        for p in sorted(base.iterdir()):
            if not p.is_dir():continue
            # Only named competition-owned projects, not other users' content.
            if not any(t in p.name.lower() for t in ['logika','depth-20261017','exploration-12h','submission-',
                    'horizontal-','calibration-','ranking-','teammate-','risk-','coverage-transfer']):continue
            if p.name=='logika-meta-experiments':
                for campaign in ['meta-20261016-v1','meta-20261016-v1-wave2']:
                    q=p/campaign
                    if not q.exists():missing.append(str(q));continue
                    db=q/'ledger.sqlite'
                    if db.exists():
                        connection=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
                        connection.row_factory=sqlite3.Row
                        tasks=[]
                        for row in connection.execute('SELECT * FROM tasks ORDER BY id'):
                            row=dict(row);row['task']=json.loads(row['task']);tasks.append(row)
                        connection.close()
                        out=dest/(campaign+'-tasks.json.gz')
                        with gzip.open(out,'wt',compresslevel=4) as f:json.dump(tasks,f,separators=(',',':'))
                        manifest.append(dict(source=str(db),category='legacy_ledger',host=host,rows=len(tasks),
                            source_sha=sha(db),snapshot=str(out.relative_to(ROOT)),snapshot_sha=sha(out),
                            states={state:sum(row['state']==state for row in tasks) for state in sorted({r['state'] for r in tasks})}))
                    for f in (q/'reports').glob('*.json'):capture(f,'legacy_report')
                    for name in ['campaign.json','campaign_status.json','source_manifest.json','catalogue.json','design_summary.json']:
                        capture(q/name,'legacy_metadata')
                    for f in (q/'plans').glob('*.json'):capture(f,'legacy_catalogue')
                continue
            for name in ['plan.json','state.json','status.json','result.json','scores.json','preflight.json','INDEPENDENT_AUDIT.json',
                'independent_verification.json','gate_result.json','environment.json','source_manifest.json','PUBLIC_EVIDENCE.json',
                'frozen_recipe.json','delivery.json','benchmark_result.json','review_summary.json','verification.json',
                'clock.json','deployment.json','input.json','phase.json','finished.json']:
                capture(p/name,'study_metadata')
            for f in (p/'reports').glob('*.json'):capture(f,'study_report')
            for name in ['configurations.json','views.json','champion.json','anchors.json']:
                capture(p/'source'/name,'study_configuration')
            for name in ['campaign','campaign-parallel']:
                for fn in ['config.json','state.json','seal.json','environment.json']:
                    capture(p/name/fn,'campaign_'+fn.split('.')[0])
            for folder in ['evidence','results/analysis','feature-study','results','PC3-results']:
                d=p/folder
                if not d.exists():continue
                for f in d.glob('*.json'):
                    if any(t in f.name.lower() for t in ['inventory','outcome','summary','manifest','prediction_audit','score','result','state','verification','audit','plan']):
                        capture(f,'additional_evidence')
            # Saved historical metric audit is reusable only after source hashes match.
            if p.name=='submission-robust-20261018':
                for name in ['all_prediction_audits.json','depth_inventory.json','depth_prediction_audits.json','latest_ensemble_inventory.json']:
                    capture(p/'evidence'/name,'prior_prediction_verification')
    for repo in repos:
        repo=Path(repo)
        if not repo.exists():missing.append(str(repo));continue
        skip={'.git','.venv','data','raw','third_party','cache','__pycache__','.ipynb_checkpoints','figures','plots'}
        for path,dirs,files in os.walk(repo):
            dirs[:]=[d for d in dirs if d not in skip and not d.startswith('.')]
            if len(Path(path).relative_to(repo).parts)>9:dirs[:]=[];continue
            for name in files:
                if name.endswith('.json') and any(t in name.lower() for t in ['result','score','metric','summary','manifest','experiment',
                    'status','report','baseline','config','definition','catalogue','verification','audit','plan']):
                    capture(Path(path)/name,'teammate_repo_history')
    dump(dest/'manifest.json',dict(host=host,created=time.time(),sources=manifest,missing_optional_sources=missing,
        no_new_model_training=True,no_existing_campaign_mutations=True,source_revisions_not_double_counted=True))
    print(json.dumps(dict(host=host,sources=len(manifest),bytes=sum((ROOT/s['snapshot']).stat().st_size for s in manifest),
                         missing_optional=len(missing))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True);parser.add_argument('--base',action='append',required=True)
    parser.add_argument('--repo',action='append',default=[])
    args=parser.parse_args();snapshot(args.host,args.base,args.repo)
