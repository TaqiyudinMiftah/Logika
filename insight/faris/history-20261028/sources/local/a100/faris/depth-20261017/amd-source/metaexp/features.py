"""Read-only raw inputs -> row-local features and fold-fitted cohort statistics."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .common import FORBIDDEN,SOURCES,sha,write,identity

RAW_NAMES=['train_applications','test_applications','customer_profiles','previous_applications',
           'repayment_history','account_behavior','bureau_observations','ecommerce_transactions']
WINDOWS=[30,60,90,180,365,'all']

def lineage(name):
    if name in ['loan_amount','term_months','annual_interest_rate','proposed_monthly_installment','loan_purpose','decision_delay_hours']:return ['application']
    if name in ['installment_income_ratio','loan_income_ratio']:return ['application','profile']
    if name=='income_per_dependent':return ['profile']
    # Conservative multi-source attribution: an uncertain dependency must not
    # survive an ablation of one of its possible parent sources.
    if any(t in name for t in ['_ratio','_cover','_surplus','_deficit','_per_dependent']):return list(SOURCES)
    if any(t in name for t in ['repay','late_','payment_delay','paid_','unpaid_']):return ['repayment']
    if 'bureau' in name:return ['bureau']
    if any(t in name for t in ['ecom','shopping','category_price','basket']):return ['shopping']
    if any(t in name for t in ['account','balance','inflow','outflow','income_like','low_balance']):return ['account']
    if any(t in name for t in ['previous','existing_active','prev_']):return ['previous']
    if name in ['age','region','employment_type','employment_tenure_months','declared_monthly_income','dependents','housing_status']:return ['profile']
    if any(t in name for t in ['income','dependent','surplus','deficit']):return ['application','profile','account']
    return list(SOURCES)

def prepare_bank(root,raw,cached):
    root=Path(root);raw=Path(raw);cached=Path(cached);out=root/'bank';out.mkdir(exist_ok=True)
    expected={n:sha(raw/(n+'.csv')) for n in RAW_NAMES}
    cached_manifest=root/'inputs/cached_manifest.json'
    if cached_manifest.exists():
        recorded=json.loads(cached_manifest.read_text())['identity']['raw_sha256']
        for n,v in expected.items():
            if recorded[n+'.csv']!=v:raise ValueError('Cached raw provenance differs: '+n)
    if (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text())
        if m['raw_sha256']!=expected:raise ValueError('Different raw inputs; new campaign required')
        for f,h in m['files'].items():
            if sha(out/f)!=h:raise ValueError('Corrupt feature bank '+f)
        return m
    train=pd.read_csv(raw/'train_applications.csv');test=pd.read_csv(raw/'test_applications.csv')
    if 'target' in test or set(train.target)!={0,1}:raise ValueError('Label boundary differs')
    base=pd.concat([train.drop(columns='target'),test],ignore_index=True)
    if not base.application_id.is_unique or not base.customer_id.is_unique:raise ValueError('Repeated application/customer')
    full=pd.read_parquet(cached/'application_features_v3.parquet')
    if full.application_id.tolist()!=base.application_id.tolist():raise ValueError('Cache row order differs')
    idx=pd.Index(base.customer_id,name='customer_id');full=full.set_index('customer_id').reindex(idx)
    numeric={};info={};cats={};priors=[];views={}
    def add(name,values,sources,window='all',tags=('raw',),**extra):
        if name in numeric:raise ValueError('Duplicate feature '+name)
        s=pd.Series(values,index=idx) if not isinstance(values,pd.Series) else values.reindex(idx)
        numeric[name]=s.replace([np.inf,-np.inf],np.nan).astype(np.float32)
        info[name]={'sources':sources,'window':window,'tags':list(tags),**extra}
    for col in full:
        if col in FORBIDDEN:continue
        if pd.api.types.is_numeric_dtype(full[col]):add(col,full[col],lineage(col),legacy=True)
        elif col not in ['application_datetime','decision_datetime']:cats[col]=full[col].astype('string').fillna('__MISSING__')
    for view,file in [('base','application_features.parquet'),('extended','application_features_extended.parquet'),('v3','application_features_v3.parquet')]:
        import pyarrow.parquet as pq
        cols=pq.read_schema(cached/file).names
        views[view]=[c for c in cols if c not in FORBIDDEN and c not in ['application_datetime','decision_datetime']]
    recipe=json.loads((root/'vendor/v3_recipe.json').read_text())
    views['v3']=next(iter(recipe['definitions'].values()))['columns']
    cutoff=base[['customer_id','decision_datetime']].copy();cutoff['decision_datetime']=pd.to_datetime(cutoff.decision_datetime)
    def history(name,timecol,other_dates=()):
        d=pd.read_csv(raw/(name+'.csv'),parse_dates=[timecol,*other_dates])
        d=d.merge(cutoff,on='customer_id',how='left',validate='many_to_one')
        if d.decision_datetime.isna().any():raise ValueError('Orphan history '+name)
        if not d[timecol].lt(d.decision_datetime).all():raise ValueError('Future history '+name)
        d['age']=(d.decision_datetime-d[timecol]).dt.total_seconds()/86400
        return d
    def stats(d,columns,prefix,source,window,tags=('raw',),functions=('mean','std','min','max','median')):
        z=d.groupby('customer_id')[columns].agg(list(functions))
        for col,fn in z.columns:add(prefix+'_'+col+'_'+fn,z[(col,fn)],[source],window,tags)
    def pieces(d):
        for w in WINDOWS:yield w,d if w=='all' else d[d.age.le(w)]
    rep=history('repayment_history','due_date',['last_payment_datetime'])
    if (rep.last_payment_datetime.notna()&rep.last_payment_datetime.ge(rep.decision_datetime)).any():raise ValueError('Future allocation')
    rep['unpaid_fraction']=(1-rep.amount_paid/rep.amount_due.where(rep.amount_due.gt(0))).clip(0,1)
    for w,d in pieces(rep):
        pre=f'm_repayment_{w}';g=d.groupby('customer_id')
        add(pre+'_count',g.size(),['repayment'],w)
        stats(d,['days_past_due','max_days_past_due','unpaid_fraction'],pre,'repayment',w,('raw','robust'))
        for threshold in [1,7,30,60,90]:
            for mature in [False,True]:
                e=d[d.age.ge(threshold)] if mature else d
                tmp=e.assign(late=e.max_days_past_due.ge(threshold).astype(float))
                group=tmp.groupby('customer_id').late.agg(['sum','count'])
                stem=pre+f'_t{threshold}_m{int(mature)}'
                k,n=stem+'_k',stem+'_n'
                extra=dict(threshold=threshold,mature=mature)
                add(k,group['sum'],['repayment'],w,('rates','posterior','uncertainty'),**extra)
                add(n,group['count'],['repayment'],w,('rates','posterior','uncertainty'),**extra)
                priors.append({'k':k,'n':n,'name':stem+'_posterior','sd':stem+'_uncertainty','source':'repayment','window':w,'threshold':threshold,'mature':mature})
                loan=tmp.groupby(['customer_id','loan_id']).late.mean().groupby('customer_id').mean()
                add(stem+'_loan_rate',loan,['repayment'],w,('loan_weighted',),**extra)
        for hl in [30,90,180,365]:
            d=d.assign(w=np.exp(-np.log(2)*d.age/hl))
            for t in [1,7,30,60,90]:
                z=d.assign(wlate=d.w*d.max_days_past_due.ge(t)).groupby('customer_id')[['w','wlate']].sum()
                add(pre+f'_decay{hl}_late{t}',z.wlate/z.w.where(z.w.gt(0)),['repayment'],w,('decay',),half_life=hl,threshold=t)
    del rep;gc.collect();print('Repayment bank complete',flush=True)
    # Replace all historical v3 cohort priors with row counts. Priors are fitted later.
    for name in list(numeric):
        m=re.fullmatch(r'v3_repay_(all|\d+d)_late_(\d+)_posterior_(\d+)',name)
        if m:
            w='all' if m[1]=='all' else int(m[1][:-1]);t=int(m[2]);s=int(m[3])
            k=f'm_repayment_{w}_t{t}_m0_k';n=f'm_repayment_{w}_t{t}_m0_n'
            priors.append(dict(k=k,n=n,name=name,strength=s,legacy=True,source='repayment',window=w,threshold=t,mature=False))
    bureau=history('bureau_observations','available_datetime',['report_datetime'])
    if bureau.report_datetime.ge(bureau.decision_datetime).any():raise ValueError('Future bureau report')
    bureau['age']=(bureau.decision_datetime-bureau.report_datetime).dt.total_seconds()/86400
    for w,d in pieces(bureau):
        pre=f'm_bureau_{w}';stats(d,['score','utilization','outstanding_balance','active_accounts','inquiries_last_6m','delinquent_accounts_last_12m'],pre,'bureau',w,('raw','robust'))
        g=d.groupby('customer_id').score
        for q in [.1,.9]:add(pre+f'_score_q{int(q*100)}',g.quantile(q),['bureau'],w,('robust',))
        latest=d.sort_values('report_datetime').drop_duplicates(['customer_id','bureau_provider'],keep='last')
        z=latest.pivot(index='customer_id',columns='bureau_provider',values='score')
        for c in z:add(pre+'_provider_'+str(c),z[c],['bureau'],w,('provider','latest'))
        add(pre+'_provider_equal',z.mean(axis=1),['bureau'],w,('provider',))
        add(pre+'_disagreement',z.max(axis=1)-z.min(axis=1),['bureau'],w,('disagreement',))
        add(pre+'_provider_coverage',z.notna().sum(axis=1),['bureau'],w,('disagreement',))
        for hl in [30,90,180,365]:
            dd=d.assign(w=np.exp(-np.log(2)*d.age/hl).where(d.score.notna(),0));dd['ws']=dd.w*dd.score.fillna(0)
            z=dd.groupby('customer_id')[['w','ws']].sum()
            add(pre+f'_decay{hl}',z.ws/z.w.where(z.w.gt(0)),['bureau'],w,('decay',),half_life=hl)
    del bureau;gc.collect();print('Bureau bank complete',flush=True)
    acc=history('account_behavior','month')
    if (acc.month+pd.offsets.MonthBegin(1)>acc.decision_datetime).any():raise ValueError('Incomplete account month')
    acc['low_balance_fraction']=acc.low_balance_days/acc.month.dt.days_in_month
    acc['surplus']=acc.income_like_credits-acc.outflows
    acc['savings_rate']=acc.surplus/acc.income_like_credits.where(acc.income_like_credits.gt(0))
    for w,d in pieces(acc):
        pre=f'm_account_{w}';cols=['inflows','outflows','income_like_credits','average_balance','minimum_balance','low_balance_fraction','surplus','savings_rate']
        stats(d,cols,pre,'account',w,('raw','robust','coverage','volatility'))
        g=d.groupby('customer_id');m=g[cols].mean();sd=g[cols].std()
        for c in cols:add(pre+'_'+c+'_cv',sd[c]/m[c].abs().where(m[c].abs().gt(0)),['account'],w,('volatility',))
        for c in ['income_like_credits','outflows','average_balance','surplus']:
            dd=d[['customer_id','age',c]].dropna().copy();dd['xy']=dd.age*dd[c];dd['xx']=dd.age**2
            z=dd.groupby('customer_id').agg(x=('age','mean'),y=(c,'mean'),xy=('xy','mean'),xx=('xx','mean'))
            add(pre+'_'+c+'_slope',(z.xy-z.x*z.y)/(z.xx-z.x**2).where((z.xx-z.x**2).gt(0)),['account'],w,('trend',))
        add(pre+'_deficit_share',d.assign(deficit=d.surplus.lt(0).astype(float)).groupby('customer_id').deficit.mean(),['account'],w,('persistence','coverage'))
        add(pre+'_low_balance_persistence',d.assign(low=d.low_balance_fraction.gt(.25).astype(float)).groupby('customer_id').low.mean(),['account'],w,('persistence',))
    del acc;gc.collect();print('Account bank complete',flush=True)
    ecom=history('ecommerce_transactions','transaction_datetime')
    if ecom.groupby('order_id').order_total.nunique().max()!=1:raise ValueError('Inconsistent order grain')
    ecom['unit_price']=ecom.item_total/ecom.quantity.where(ecom.quantity.gt(0))
    for w,d in pieces(ecom):
        pre=f'm_shopping_{w}';completed=d[d.order_status.eq('completed')]
        z=completed.groupby(['customer_id','product_category']).agg(price=('unit_price','median'),price_mean=('unit_price','mean'),spend=('item_total','sum'),quantity=('quantity','mean')).unstack('product_category')
        for stat,cat in z.columns:add(pre+'_'+str(cat)+'_'+stat,z[(stat,cat)],['shopping'],w,('prices' if stat.startswith('price') else 'composition',),category=str(cat),price=stat.startswith('price'))
        prices=z['price'];overall=np.exp(np.log(prices.where(prices.gt(0))).mean(axis=1))
        for cat in prices:add(pre+'_'+str(cat)+'_relative_price',prices[cat]/overall,['shopping'],w,('prices','composition'))
        spend=z['spend'].fillna(0);shares=spend.div(spend.sum(axis=1).where(spend.sum(axis=1).gt(0)),axis=0)
        add(pre+'_entropy',-(shares.where(shares.gt(0))*np.log(shares.where(shares.gt(0)))).sum(axis=1),['shopping'],w,('composition',))
        orders=d.drop_duplicates('order_id');orders=orders.assign(cancelled=orders.order_status.eq('cancelled').astype(float))
        add(pre+'_cancelled_fraction',orders.groupby('customer_id').cancelled.mean(),['shopping'],w,('cancelled',))
        add(pre+'_cancelled_basket_value',orders[orders.cancelled.eq(1)].groupby('customer_id').order_total.mean(),['shopping'],w,('cancelled',))
        paid=orders[orders.cancelled.eq(0)];stats(paid,['order_total'],pre+'_orders','shopping',w,('raw','prices'))
        pay=pd.crosstab(paid.customer_id,paid.payment_method).astype(float);pay=pay.div(pay.sum(axis=1),axis=0)
        for method in pay:add(pre+'_payment_'+str(method),pay[method],['shopping'],w,('payment',))
        monthly=paid.assign(bucket=(paid.age//30).astype(int)).groupby(['customer_id','bucket']).order_total.sum().unstack(fill_value=0)
        add(pre+'_monthly_cv',monthly.std(axis=1)/monthly.mean(axis=1).where(monthly.mean(axis=1).gt(0)),['shopping'],w,('seasonality',))
    del ecom;gc.collect();print('Shopping bank complete',flush=True)
    prev=history('previous_applications','decision_datetime') if False else pd.read_csv(raw/'previous_applications.csv',parse_dates=['decision_datetime','application_datetime'])
    prev=prev.rename(columns={'decision_datetime':'historical_decision'}).merge(cutoff,on='customer_id',validate='many_to_one')
    if prev.historical_decision.ge(prev.decision_datetime).any():raise ValueError('Future prior decision')
    prev['age']=(prev.decision_datetime-prev.historical_decision).dt.total_seconds()/86400
    for w,d in pieces(prev):
        pre=f'm_previous_{w}';g=d.groupby('customer_id')
        add(pre+'_count',g.size(),['previous'],w,('contracts',))
        for decision in ['approved','rejected','cancelled']:
            add(pre+'_'+decision+'_share',d.assign(flag=d.decision.eq(decision).astype(float)).groupby('customer_id').flag.mean(),['previous'],w,('contracts',))
        stats(d,['requested_amount','approved_amount','scheduled_monthly_installment'],pre,'previous',w,('contracts',))
        add(pre+'_recent_age',g.age.min(),['previous'],w,('contracts',))
    del prev;gc.collect()
    # Domain-constrained interactions: ratios have clear units and safe denominators.
    fields=['loan_amount','proposed_monthly_installment','declared_monthly_income','income_per_dependent',
            'v2_account_3m_income_like_credits_mean','v2_account_3m_outflows_mean','v2_account_3m_average_balance_mean','v2_existing_active_installment']
    fields=[c for c in fields if c in numeric]
    for i,a in enumerate(fields):
        for b in fields[i+1:]:
            sources=sorted(set(info[a]['sources']+info[b]['sources']))
            add('m_cross_'+a+'__over__'+b,numeric[a]/numeric[b].where(numeric[b].gt(0)),sources,tags=('ratios','affordability'))
            add('m_cross_'+a+'__minus__'+b,numeric[a]-numeric[b],sources,tags=('income_gap','affordability'))
            add('m_cross_'+a+'__times__'+b,np.log1p(numeric[a].clip(lower=0))*np.log1p(numeric[b].clip(lower=0)),sources,tags=('products',))
    columns=list(numeric);matrix=np.lib.format.open_memmap(out/'numeric.npy',mode='w+',dtype='float32',shape=(len(idx),len(columns)))
    for j,c in enumerate(columns):matrix[:,j]=numeric[c].to_numpy()
    matrix.flush();del matrix,numeric;gc.collect()
    catframe=pd.DataFrame(cats,index=idx).reset_index(drop=True);catframe.to_parquet(out/'categorical.parquet',index=False)
    metadata=base[['application_id','customer_id','decision_datetime']].copy();metadata['split']=['train']*len(train)+['test']*len(test)
    metadata.to_parquet(out/'metadata.parquet',index=False)
    from sklearn.model_selection import train_test_split,StratifiedKFold
    strata=train.target.astype(str)+'_'+pd.to_datetime(train.decision_datetime).dt.strftime('%Y-%m')
    dev,confirm=train_test_split(np.arange(len(train)),test_size=.2,stratify=strata,random_state=20261016)
    dev=np.sort(dev);confirm=np.sort(confirm);devy=train.target.to_numpy(dtype=np.int8)[dev]
    folds=np.empty(len(dev),dtype=np.int8)
    for f,(_,vi) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=9173).split(dev,devy)):folds[vi]=f
    np.save(out/'development_rows.npy',dev);np.save(out/'confirmation_rows.npy',confirm)
    np.save(out/'development_y.npy',devy);np.save(out/'confirmation_y.npy',train.target.to_numpy(dtype=np.int8)[confirm]);np.save(out/'folds.npy',folds)
    split=pd.DataFrame({'application_id':metadata.application_id.iloc[dev].to_numpy(),'fold':folds});split.to_csv(out/'splits.csv',index=False)
    manifest={'raw_sha256':expected,'columns':columns,'categorical_columns':list(cats),'info':info,'priors':priors,'views':views,
              'shape':[len(idx),len(columns)],'train_rows':len(train),'test_rows':len(test),'development_rows':len(dev),'confirmation_rows':len(confirm),
              'files':{},'historical_selection_caveat':'Historical feature definitions used these labels; confirmation is reserved only for this campaign.'}
    manifest['files']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    manifest['bank_id']=identity(manifest);write(out/'manifest.json',manifest)
    print(json.dumps({'bank_id':manifest['bank_id'],'shape':manifest['shape'],'development':len(dev),'confirmation':len(confirm)}),flush=True)
    return manifest

class Bank:
    def __init__(self,root):
        self.root=Path(root)/'bank';self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.names=self.manifest['columns'];self.lookup={c:i for i,c in enumerate(self.names)}
        self.numeric=np.load(self.root/'numeric.npy',mmap_mode='r');self.categorical=pd.read_parquet(self.root/'categorical.parquet')
        self.dev=np.load(self.root/'development_rows.npy');self.y=np.load(self.root/'development_y.npy');self.folds=np.load(self.root/'folds.npy')
        self.metadata=pd.read_parquet(self.root/'metadata.parquet')
    def columns(self,representation):
        c=representation['config'];family=representation['family'];sources=set(c['sources']);view=self.manifest['views'][c['view']]
        names=[n for n in view if n in self.lookup and set(self.manifest['info'][n]['sources'])<=sources]
        domains={'repayment':'repayment','bureau':'bureau','account':'account','shopping':'shopping','cross_source':'cross'}
        if family in domains:
            extra=[]
            for n in self.names:
                inf=self.manifest['info'][n]
                if not n.startswith('m_'+domains[family]+'_'):continue
                if not set(inf['sources'])<=sources:continue
                if inf['window'] not in [c['window'],'all'] and not n.startswith('m_cross_'):continue
                if 'half_life' in inf and inf['half_life']!=c['half_life']:continue
                if 'threshold' in inf and inf['threshold']!=c['threshold']:continue
                if 'mature' in inf and inf['mature']!=c['mature']:continue
                if c['representation']!='raw' and c['representation'] not in inf['tags']:continue
                extra.append(n)
            if family=='cross_source':extra += [n for n in self.names if n.startswith('m_previous_') and self.manifest['info'][n]['window'] in [c['window'],'all']]
            names=(names if c['append'] else [])+extra
        if c['interactions']:names += [n for n in self.names if n.startswith('m_cross_') and set(self.manifest['info'][n]['sources'])<=sources]
        names=list(dict.fromkeys(names))
        cats=[n for n in self.manifest['categorical_columns'] if n in view and set(lineage(n))<=sources]
        if not names and not cats:raise ValueError('Empty source feature view')
        return names,cats
    def raw(self,rows,names):return np.array(self.numeric[np.asarray(rows)[:,None],np.array([self.lookup[n] for n in names])[None,:]],dtype=np.float32)
    def cohort(self,fitrows,rows,names,c):
        x=self.raw(rows,names);lookup={n:j for j,n in enumerate(names)}
        for p in self.manifest['priors']:
            if p.get('legacy'):
                if p['name'] not in lookup:continue
                pos=lookup[p['name']];k=self.numeric[fitrows,self.lookup[p['k']]];n=self.numeric[fitrows,self.lookup[p['n']]]
                prior=float(np.nansum(k)/max(np.nansum(n),1));kr=self.numeric[rows,self.lookup[p['k']]];nr=self.numeric[rows,self.lookup[p['n']]]
                s=p['strength'];x[:,pos]=(kr+s*prior)/(nr+s)
        if c.get('category_normalize'):
            for name,j in lookup.items():
                if self.manifest['info'][name].get('price'):
                    v=self.numeric[fitrows,self.lookup[name]];mu=np.nanmedian(v)
                    if np.isfinite(mu) and mu>0:x[:,j]/=mu
        # Add posterior and uncertainty columns for selected count pairs.
        extra=[];extra_names=[]
        for p in self.manifest['priors']:
            if p.get('legacy') or p['k'] not in lookup or p['n'] not in lookup:continue
            k=self.numeric[fitrows,self.lookup[p['k']]];n=self.numeric[fitrows,self.lookup[p['n']]]
            prior=float(np.nansum(k)/max(np.nansum(n),1));kr=x[:,lookup[p['k']]];nr=x[:,lookup[p['n']]];s=c['strength']
            rate=np.divide(kr+s*prior,nr+s,out=np.full(len(rows),np.nan,dtype=np.float32),where=(nr+s)>0)
            representation=c.get('representation','raw')
            if representation=='rates':
                extra.append(rate);extra_names.append(p['name'].replace('_posterior','_rate'))
            elif representation=='posterior':extra.append(rate);extra_names.append(p['name'])
            else:
                extra.extend([rate,np.sqrt(np.maximum(rate*(1-rate)/(nr+s+1),0))]);extra_names.extend([p['name'],p['sd']])
        if extra:x=np.column_stack([x,*extra])
        return x,names+extra_names
