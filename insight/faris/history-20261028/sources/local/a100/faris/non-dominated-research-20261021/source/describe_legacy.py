"""Resolve legacy catalogue labels to executed input/transform semantics.

No fitting. Descriptors describe algorithms, not fitted coefficients or seeds.
Unknown later formats remain explicit rather than acquiring invented novelty.
"""
from pathlib import Path
import ast,json,gzip,hashlib,collections,importlib.util
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
SOURCE=REPO/'a100/faris/logika-meta-experiments/metaexp/features.py'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def main():
    # Extract exactly the pure column-selection functions from the inspected
    # implementation, avoiding feature-array loads or learner imports.
    tree=ast.parse(SOURCE.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='lineage':nodes.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='Bank':
            nodes.extend(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='columns')
    assert len(nodes)==2
    ns={'SOURCES':['application','profile','previous','repayment','bureau','account','shopping']}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    bank=type('MetadataBank',(),{})();bank.manifest=read(ROOT/'snapshots/bank-manifest.json')
    bank.names=bank.manifest['columns'];bank.lookup={n:i for i,n in enumerate(bank.names)}
    archived=read(ROOT/'inventory/retained_records.json.gz');out=[];unresolved=[];cache={}
    for row in archived:
        config=row.get('configuration')
        if row['kind']!='pipeline' or not row['id'].startswith('meta/') or not config or 'representation' not in config:
            unresolved.append(dict(id=row['id'],reason='Not a legacy fitted pipeline; requires its own executed-source adapter.'));continue
        rep=config['representation'];c=rep['config'];model=config['model'];key=digest([rep['family'],c,model])
        if key not in cache:
            names,cats=ns['columns'](bank,rep);derived=[];priors=[]
            for p in bank.manifest['priors']:
                if p.get('legacy') and p['name'] in names:
                    priors.append('fold_prior:'+p['name']+':strength='+str(p['strength']))
                elif not p.get('legacy') and p['k'] in names and p['n'] in names:
                    priors.append('fold_prior:'+p['name']+':strength='+str(c['strength']))
                    if c['representation']=='rates':derived.append(p['name'].replace('_posterior','_rate'))
                    else:derived.append(p['name'])
                    if c['representation'] not in ['rates','posterior']:derived.append(p['sd'])
            selection=c['select']
            if selection=='gain':selection='lightgbm_split_count_60trees_15leaves'
            if selection=='none' and c['feature_count']!='all':selection='variance'
            prep=['imputation:'+c['missing'],'drop_training_constant_columns',
                'numeric_transform:'+c['transform'],
                'scaling:'+('robust' if c['scale']=='robust' else 'standard' if c['scale']!='none' or model=='linear' or c['reducer']!='none' else 'none')]
            if cats:prep.append('categorical:'+c['encoding']+(':inner3_smoothing20' if c['encoding']=='target' else ''))
            if c['indicator']:prep.append('fold_missing_indicators')
            if c['clip']:prep.append('fold_quantile_clip:'+str(c['clip']))
            if c.get('category_normalize') and any(bank.manifest['info'][n].get('price') for n in names):prep.append('training_median_price_normalization')
            if selection!='none':
                prep.append('selector:'+selection+':count='+str(c['feature_count']))
                if selection=='correlation':prep.append('absolute_correlation_cutoff:'+str(c['correlation']))
            representation=['numeric_columns','fold_categorical_encoding']
            if c['reducer']!='none':representation.append(c['reducer']+':dimensions='+str(min(c['dimensions'],8) if c['reducer']=='pls' else c['dimensions'])+(':append' if c['append'] else ':replace'))
            if c['spline']:representation.append('spline_degree3_knots5_first64_append')
            parameters={'lgb':{'trees':250,'leaves':31,'rate':.035,'minimum':200,'l2':40,'bins':63},
                'cat':{'trees':300,'depth':5,'rate':.035,'l2':25},
                'xgb':{'trees':300,'depth':3,'rate':.035,'minimum_hessian':80,'l2':50,'bins':128},
                'linear':{'C':.05,'max_iterations':200,'tol':.001}}[model]
            training=[k+'='+str(v) for k,v in sorted(parameters.items())]
            training+=['class_weight='+str(c['weight'])]
            if c.get('cohort_weight'):training.append('train_quantile_cohort_weights_first_column_clipped_0.25_4')
            design=dict(model_family=model,architecture_stages=[{'lgb':'leafwise_boosted_trees','cat':'symmetric_boosted_trees',
                'xgb':'depthwise_boosted_trees','linear':'logistic_regression'}[model]],
                representations=sorted(representation),features=sorted(set(names+cats+derived)),
                preprocessing_stages=sorted(set(prep+priors)),training_objective='binary_logloss',
                training_parameters=training,inference_stages=['positive_class_probability','remove_class_prior_weight_shift'])
            cache[key]=design
        design=cache[key]
        out.append(dict(id=row['id'],design=design,design_id=digest(design),cohort=row['cohort'],status=row['status'],
            reported_metrics=row['reported_metrics'],scientific_rank_eligible=False))
    by=collections.defaultdict(list)
    for row in out:by[row['design_id']].append(row['id'])
    (ROOT/'descriptors').mkdir(exist_ok=True)
    with gzip.open(ROOT/'descriptors/legacy_resolved.json.gz','wt',compresslevel=5) as f:json.dump(out,f,separators=(',',':'))
    summary=dict(records=len(out),semantic_designs=len(by),repeat_records=sum(len(ids)-1 for ids in by.values()),
        unresolved_records=len(unresolved),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        bank_id=bank.manifest['bank_id'],scope='Legacy single-pipeline recipes only; no numeric scores ranked yet.',
        effective_numeric_note='Fixed learner settings remain explicit algorithm parameters. Inactive generator factors are absent.',
        importance_correction='Selector named gain is split-count importance in executed source.',
        evidence_limit='Input and algorithm descriptions, not a claim that learned feature coefficients match across folds.')
    (ROOT/'descriptors/legacy_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'descriptors/unresolved_records.json').write_text(json.dumps(unresolved,separators=(',',':'))+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
