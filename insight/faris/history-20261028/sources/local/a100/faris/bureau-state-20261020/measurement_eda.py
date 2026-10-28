"""Unlabelled bureau dynamics check before a predictive feature experiment."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
RAW = Path('/mnt/home/logika-faris-exploration/exploration-12h-20261017/raw')
INPUT = Path('/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/input')

def dump(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()

def run():
    start = time.time()
    features = pd.read_parquet(INPUT / 'data/train_features.parquet', columns=['application_id', 'customer_id'])
    context = np.load(INPUT / 'artifacts/fold0/indices.npz')['context']
    ids = features.customer_id.iloc[context].to_numpy()
    # Neither selected-application target nor saved default predictions is loaded.
    applications = pd.read_csv(RAW / 'train_applications.csv',
                               usecols=['customer_id', 'decision_datetime'], parse_dates=['decision_datetime'])
    applications = applications.set_index('customer_id').loc[ids]
    bureau = pd.read_csv(RAW / 'bureau_observations.csv',
        usecols=['application_id','customer_id','bureau_provider','score','report_datetime','available_datetime'],
        parse_dates=['report_datetime','available_datetime'])
    bureau = bureau[bureau.customer_id.isin(ids)].copy()
    bureau['cutoff'] = bureau.customer_id.map(applications.decision_datetime)
    assert (bureau.report_datetime < bureau.cutoff).all()
    assert (bureau.available_datetime < bureau.cutoff).all()
    bureau = bureau[bureau.score.notna()].copy()
    # Entire customers are split for the measurement-prediction check.
    rng = np.random.default_rng(20261024)
    fitted_ids = set(rng.permutation(ids)[:25000])
    fit = bureau[bureau.customer_id.isin(fitted_ids)].copy()
    paired = fit.pivot_table(index=['customer_id','application_id'],
                            columns='bureau_provider', values='score', aggfunc='mean')
    providers = sorted(paired.columns)
    pair_variance, offsets = {}, [0.]
    for j in range(1, len(providers)):
        offsets.append(-float((paired[providers[0]] - paired[providers[j]]).mean()))
    offsets = np.asarray(offsets); offsets -= offsets.mean()
    for a in range(3):
        for b in range(a+1,3):
            pair_variance[(a,b)] = float((paired[providers[a]]-paired[providers[b]]).dropna().var(ddof=0))
    v01,v02,v12 = [pair_variance[k] for k in [(0,1),(0,2),(1,2)]]
    noise = np.maximum([(v01+v02-v12)/2,(v01+v12-v02)/2,(v02+v12-v01)/2], 1.)
    om = dict(zip(providers, offsets)); nm = dict(zip(providers, noise))
    for frame in [fit, bureau]:
        frame['adjusted'] = frame.score - frame.bureau_provider.map(om)
        frame['noise'] = frame.bureau_provider.map(nm)
        frame['precision'] = 1 / frame.noise
        frame['weighted'] = frame.adjusted * frame.precision
    summary = fit.groupby('customer_id').agg(weighted=('weighted','sum'), precision=('precision','sum'))
    means = summary.weighted / summary.precision
    prior_mean = float(means.mean())
    prior_variance = max(float(means.var(ddof=0) - np.mean(1/summary.precision)), 1.)
    # Same-provider differences remove provider shifts. Actual elapsed report time
    # drives the process term; selected default labels play no role.
    f = fit.sort_values(['customer_id','bureau_provider','report_datetime'])
    group = f.groupby(['customer_id','bureau_provider'])
    difference = group.adjusted.diff().to_numpy()
    lag = group.report_datetime.diff().dt.total_seconds().to_numpy()/86400
    variance = 2*f.noise.to_numpy()
    valid = np.isfinite(difference) & np.isfinite(lag) & (lag>0)
    difference,lag,variance = difference[valid],lag[valid],variance[valid]
    def objective(q):
        vv = variance + q*lag
        return float(np.mean(np.log(vv) + difference**2/vv)/2)
    optimum = minimize_scalar(objective, bounds=(0.,100.), method='bounded', options={'xatol':1e-7})
    process = float(optimum.x) if optimum.fun < objective(0.) else 0.
    bins=[]
    for lo,hi in [(0,30),(30,90),(90,180),(180,365),(365,730),(730,3000)]:
        mask=(lag>lo)&(lag<=hi)
        if mask.any():
            bins.append(dict(lag_days=[lo,hi],pairs=int(mask.sum()),
                mean_squared_difference=float(np.mean(difference[mask]**2)),
                expected_measurement_variance=float(np.mean(variance[mask])),
                median_difference=float(np.median(difference[mask]))))
    # Sequential prediction of the final held-out bureau score per customer,
    # with all earlier measurements available. Selection defaults are not used.
    check = bureau[~bureau.customer_id.isin(fitted_ids)].copy()
    check = check.sort_values(['customer_id','available_datetime','report_datetime','bureau_provider'])
    rows=[]
    for cid, part in check.groupby('customer_id',sort=False):
        if len(part)<2: continue
        value=part.adjusted.to_numpy();measurement=part.noise.to_numpy()
        # Availability time preserves the information order. Report age is retained
        # as measurement variance, rather than updating backward in information time.
        day=part.available_datetime.astype('int64').to_numpy()/8.64e13
        static_mean=dynamic_mean=prior_mean
        static_var=dynamic_var=prior_variance
        previous=day[0]
        for j in range(len(part)):
            dynamic_var += process*max(day[j]-previous,0.)
            if j==len(part)-1:
                rows.append([value[j]-static_mean,value[j]-dynamic_mean,
                             static_var+measurement[j],dynamic_var+measurement[j],len(part)])
            gain=static_var/(static_var+measurement[j])
            static_mean+=gain*(value[j]-static_mean);static_var*=1-gain
            gain=dynamic_var/(dynamic_var+measurement[j])
            dynamic_mean+=gain*(value[j]-dynamic_mean);dynamic_var*=1-gain
            previous=day[j]
    z=np.asarray(rows)
    result=dict(status='complete',selected_default_labels_used=False,context_rows=50000,
        parameter_fit_customers=25000,measurement_check_customers=len(rows),
        provider_offsets=om,provider_noise_variances=nm,prior_mean=prior_mean,prior_variance=prior_variance,
        estimated_process_variance_per_day=process,monthly_process_sd=float(np.sqrt(process*30)),
        difference_likelihood_static=objective(0.),difference_likelihood_dynamic=objective(process),
        variogram=bins,measurement_check={
            'static_MSE':float(np.mean(z[:,0]**2)), 'dynamic_MSE':float(np.mean(z[:,1]**2)),
            'static_gaussian_NLL':float(np.mean(np.log(z[:,2])+z[:,0]**2/z[:,2])/2),
            'dynamic_gaussian_NLL':float(np.mean(np.log(z[:,3])+z[:,1]**2/z[:,3])/2)},
        caveat='A measurement forecast is not a default score. Provider independence, Brownian evolution and availability ordering are testable approximations.',
        source_sha=sha(__file__),input_sha={n:sha(RAW/n) for n in ['bureau_observations.csv','train_applications.csv']},
        elapsed_seconds=time.time()-start)
    dump('MEASUREMENT_EDA.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
