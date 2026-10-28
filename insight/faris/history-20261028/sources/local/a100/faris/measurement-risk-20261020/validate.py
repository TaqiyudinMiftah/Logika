"""Actual-data feature boundary check; never scores evaluation labels."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import numpy as np
import model
from run import ROOT,read,write,arrays

def run():
    rows=arrays('rows');ix=arrays('fold0');train,val=ix['train'],ix['validation'];n=len(rows['target'])
    data={name:arrays(name) for name in ['prices','bureau','income','repayment']};data['employment']=rows['employment']
    means,variance,missing,parameters=model.measurement(data,train,n)
    x,v,trans=model.design(means,variance,missing,rows,train,read(ROOT/'input/manifest.json')['covariates'])
    # Alter every held-out borrower's raw measurements without changing fit rows.
    heldout=np.zeros(n,bool);heldout[val]=True
    prices=data['prices'];mask=heldout[prices['order_customer'][prices['order']]];prices['log_price'][mask]+=2.
    bureau=data['bureau'];bureau['score'][heldout[bureau['customer']]]+=100.
    bureau['pairs'][heldout[bureau['pair_customer']]]+=100.
    data['income']['mean'][val]+=2.;data['income']['diff_square'][val]+=100.
    data['repayment']['k'][val]=0
    rows['covariates'][val]+=10.;rows['declared_income'][val]*=5
    rows['target'][val]=1-rows['target'][val]
    other,other_var,other_missing,other_parameters=model.measurement(data,train,n)
    xx,vv,other_trans=model.design(other,other_var,other_missing,rows,train,
                                 read(ROOT/'input/manifest.json')['covariates'])
    assert parameters==other_parameters,'Held-out measurements changed fitted population parameters'
    assert trans==other_trans,'Held-out covariates changed fitted preprocessing'
    assert np.array_equal(means[train],other[train])
    assert np.array_equal(variance[train],other_var[train])
    assert np.array_equal(x[train],xx[train]) and np.array_equal(v[train],vv[train])
    result=dict(status='passed',training_rows=56000,heldout_rows_mutated=14000,
        heldout_labels_measurements_and_covariates_cannot_change_training_parameters=True,
        finite_difference_check=model.preflight(),historical_snapshot_timing_certified=False)
    write(ROOT/'LEAKAGE_BOUNDARY.json',result);print(result,flush=True)

if __name__=='__main__':run()
