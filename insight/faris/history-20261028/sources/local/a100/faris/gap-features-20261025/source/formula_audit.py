"""Small hand-calculated references for the new row-local summaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from features import liquidity,alignment,path_features
g=pd.DataFrame({'month':pd.to_datetime(['2025-01-01','2025-02-01','2025-03-01']),
    'cutoff':pd.Timestamp('2025-04-01'),'income_like_credits':[100,100,100],
    'inflows':[100,100,100],'outflows':[80,105,92],
    'minimum_balance':[15,3,2],'closing_balance':[40,20,25]})
actual=liquidity(g,10)
expected=[2/3,17/300,2/3,-.124,2/3,2/3,np.nan,3]
assert np.allclose(actual,expected,equal_nan=True)
g['customer_id']='example';g['spend']=[10,20,30];g['basket']=[20,20,40];g['cancel']=[10,0,10];g['credit']=[0,10,10]
a=alignment(g,10)
# All incomes equal => lowincome split and correlations undefined.
assert np.allclose(a,[60/277,np.nan,np.nan,10/60-10/20,20/50-0/10,20/60,np.nan,3],equal_nan=True)
history=pd.DataFrame({'decision_datetime':pd.to_datetime(['2024-01-02','2024-03-03','2024-03-21']),
    'application_datetime':pd.to_datetime(['2024-01-01','2024-03-01','2024-03-20']),
    'application_id':['a','b','c'],'decision':['rejected','approved','rejected'],
    'requested_amount':[100,200,300],'approved_amount':[np.nan,150,np.nan]})
app=pd.Series({'decision_datetime':pd.Timestamp('2024-04-01'),'loan_amount':400,'declared_monthly_income':100})
p=path_features(history,app)
assert np.isclose(p[0],.5) and p[2]==1 and np.isclose(p[4],np.log(201/101))
assert np.isclose(p[5],4) and np.isclose(p[6],np.log(151/201)) and p[7]==11
out={'status':'passed','checks':['hand-computed shortfall severity, fractions, calendar run and quantile','hand-computed aligned totals and low-buffer conditional fractions','chronological nonapproval, reapplication and amount transitions'],
    'labels_used':False,'prediction_or_selection_change':False}
(Path(__file__).resolve().parents[1]/'FORMULA_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
