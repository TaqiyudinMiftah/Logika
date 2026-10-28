"""Stream actual item events, preserving item/order grains, without targets.

Own-category contrasts require no population-fitted price normalization.
The CSV's customer blocks are verified; a repeated non-contiguous customer
fails rather than silently producing incomplete features. Peak event storage
is one customer's history, not the 4.36-million-row table.
"""
from pathlib import Path
from datetime import datetime
from itertools import groupby
import argparse
import csv
import hashlib
import json
import math
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
OUT=V/'data_fe'

SUPPORT=[
    'ip_price_comparable_categories', 'ip_multicat_price_orders180', 'ip_valid_price_items180',
    'ip_price_change_observed', 'ip_order_coherence_observed',
    'ip_affordability_cross_observed', 'ip_debt_cross_observed', 'ip_liquidity_cross_observed']
PRICE=['ip_category_price_change_shrunk','ip_category_price_change_breadth','ip_category_price_change_dispersion']
COHERENCE=['ip_within_order_price_covariance_shrunk','ip_within_order_price_dispersion']
CROSS=['ip_price_upgrade_x_installment_burden','ip_order_covariance_x_total_burden','ip_price_dispersion_x_liquidity_shortfall']
ALL=SUPPORT+PRICE+COHERENCE+CROSS
CATEGORIES={'groceries','fashion','electronics','home','health','education','personal_care'}


def moments(values):
    if not values:return (0.,0.,0)
    return (sum(values),sum(x*x for x in values),len(values))


def customer_features(events,cutoff,financial):
    orders={};seen_items=set();recent={};prior={};all_prices={};checks=dict(rows=0,future_events=0,nonpositive_price_rows=0,cancelled_items=0,completed_orders180=0)
    for row in events:
        checks['rows']+=1
        item=row['order_item_id']
        assert item not in seen_items,'Duplicate item within customer'
        seen_items.add(item)
        when=datetime.fromisoformat(row['transaction_datetime'])
        assert when<cutoff,'Event is not known strictly before application decision'
        age=(cutoff-when).total_seconds()/86400.
        assert row['product_category'] in CATEGORIES
        status=row['order_status'];assert status in ['completed','cancelled']
        order_key=row['order_id'];total=int(row['order_total']);item_total=int(row['item_total']);quantity=int(row['quantity'])
        signature=(when,status,row['payment_method'],total)
        if order_key not in orders:orders[order_key]=dict(signature=signature,item_total=0,items=[],age=age)
        order=orders[order_key];assert order['signature']==signature,'Inconsistent order-level fields'
        order['item_total']+=item_total
        if status=='cancelled':checks['cancelled_items']+=1;continue
        if quantity<=0 or item_total<=0:checks['nonpositive_price_rows']+=1;continue
        if age>180.:continue
        category=row['product_category'];price=math.log1p(item_total/quantity)
        all_prices.setdefault(category,[]).append(price)
        (recent if age<30. else prior).setdefault(category,[]).append(price)
        order['items'].append((category,price))
    for order in orders.values():
        assert order['item_total']==order['signature'][3],'Order total differs from sum of items'
    changes=[];weights=[]
    for category,r in recent.items():
        previous=prior.get(category,[])
        if len(previous)<2:continue
        delta=sum(r)/len(r)-sum(previous)/len(previous)
        effective=len(r)*len(previous)/(len(r)+len(previous))
        weight=effective/(effective+3.)
        changes.append(delta);weights.append(weight)
    n_categories=len(changes)
    # All three price features share this explicit eligibility definition.
    price_observed=n_categories>=2
    if price_observed:
        shrunk=np.array(changes)*weights
        price_values=[float(shrunk.mean()),float(np.dot(weights,np.array(changes)>0)/sum(weights)),float(shrunk.std())]
    else:price_values=[np.nan]*3
    means={c:sum(x)/len(x) for c,x in all_prices.items()}
    products=[];dispersion=[]
    for order in orders.values():
        if order['signature'][1]!='completed' or order['age']>180.:continue
        checks['completed_orders180']+=1
        items=order['items'];m=len(items)
        if m<2:continue
        by_category={}
        for c,lp in items:by_category.setdefault(c,[]).append(lp-means[c])
        if len(by_category)<2:continue
        all_residual=[x for values in by_category.values() for x in values]
        ss=sum(x*x for x in all_residual);s=sum(all_residual)
        pair_count=(m*(m-1)-sum(len(x)*(len(x)-1) for x in by_category.values()))/2.
        within=sum(sum(x)**2-sum(v*v for v in x) for x in by_category.values())
        cross_product=((s*s-ss)-within)/2.
        all_difference=m*ss-s*s
        same_difference=sum(len(x)*sum(v*v for v in x)-sum(x)**2 for x in by_category.values())
        cross_difference=all_difference-same_difference
        assert cross_difference>=-1e-9
        products.append(cross_product/pair_count)
        dispersion.append(max(0.,cross_difference)/(2.*pair_count))
    n_multicat=len(products);coherence_observed=n_multicat>=3
    if coherence_observed:
        coherence_values=[float(n_multicat/(n_multicat+5.)*np.mean(products)),float(np.mean(dispersion))]
    else:coherence_values=[np.nan]*2
    installment,observed_burden,total_burden,balance=financial
    affordable=price_observed and np.isfinite(observed_burden) and observed_burden>=0
    debt=coherence_observed and np.isfinite(total_burden) and total_burden>=0
    liquidity=coherence_observed and np.isfinite(balance) and np.isfinite(installment) and installment>0
    cross=[max(price_values[0],0.)*math.log1p(observed_burden) if affordable else np.nan,
           max(coherence_values[0],0.)*math.log1p(total_burden) if debt else np.nan,
           math.sqrt(coherence_values[1])*max(installment-max(balance,0.),0.)/installment if liquidity else np.nan]
    support=[n_categories,n_multicat,sum(len(x) for x in all_prices.values()),float(price_observed),float(coherence_observed),float(affordable),float(debt),float(liquidity)]
    return np.r_[support,price_values,coherence_values,cross],checks


def catalog():
    formula={
        PRICE[0]:'mean_c(w_c*delta_c), delta_c=mean(log1p(unit_price),0:30d,c)-mean(log1p(unit_price),30:180d,c); w_c=n_eff/(n_eff+3), n_eff=n_recent*n_prior/(n_recent+n_prior)',
        PRICE[1]:'sum_c(w_c*I(delta_c>0))/sum_c(w_c)',
        PRICE[2]:'population std across comparable categories of w_c*delta_c',
        COHERENCE[0]:'n_multicat/(n_multicat+5)*mean_order(mean_distinct_category_pairs(r_i*r_j)); r=log1p(unit_price)-own category mean in180d',
        COHERENCE[1]:'mean_order(mean_distinct_category_pairs((r_i-r_j)^2/2)); equal weight per order',
        CROSS[0]:'max(ip_category_price_change_shrunk,0)*log1p(installment_observed_income_ratio)',
        CROSS[1]:'max(ip_within_order_price_covariance_shrunk,0)*log1p(v2_existing_total_burden_observed)',
        CROSS[2]:'sqrt(ip_within_order_price_dispersion)*max(new_installment-max(account_average_balance3m,0),0)/new_installment'}
    support_formula=['number categories with>=1recent30d item and>=2prior30:180d items','number completed180d orders with valid prices in>=2categories','number valid completed180d item prices','I(comparable categories>=2)','I(multicat180d orders>=3)','I(price feature and observed installment burden defined)','I(coherence feature and total observed burden defined)','I(coherence feature, average balance defined and new installment>0)']
    for c,f in zip(SUPPORT,support_formula):formula[c]=f
    rows=[]
    for c in ALL:
        family='support' if c in SUPPORT else 'price_change' if c in PRICE else 'order_coherence' if c in COHERENCE else 'cross_capacity'
        rows.append(dict(feature_name=c,family=family,formula=formula[c],source_tables='ecommerce_transactions; applications; existing completed-month account aggregates for cross features',
                         cutoff_rule='All own item events strictly before selected decision; prices only completed orders in180d; prior price window30:180d; no population/time/test-label reference',
                         hypothesis='Separate within-category preference movement and within-order common price variation from basket composition; test whether extra variation matters under repayment-capacity stress',
                         missing_rule='Price values require>=2 comparable categories; coherence values require>=3multicategory orders; crosses require documented denominators; undefined remainsNaN with explicit shared support indicators',
                         leakage_risk='Incrementally low for item events; cross features inherit baseline account aggregation provenance; complete pipeline not certified',
                         caveat='No SKU or quality identifier; price changes may reflect product mix within a category. Own-category centering uses all observed180d events before the current application, not information available at each old transaction.',
                         priority='P1',validation_status='Unvalidated'))
    return rows


def preflight():
    from datetime import timedelta
    cutoff=datetime(2025,6,1);events=[]
    for order in range(8):
        recent=order<3
        when=cutoff-timedelta(days=10+order if recent else 40+order)
        for j,category in enumerate(['groceries','home']):
            value=int((100+20*order+10*j)*(1.2 if recent else 1.))
            events.append(dict(order_item_id=f'I{order}_{j}',order_id=f'O{order}',transaction_datetime=when.isoformat(sep=' '),product_category=category,quantity='1',item_total=str(value),order_total=str(sum(int((100+20*order+10*k)*(1.2 if recent else 1.)) for k in range(2))),order_status='completed',payment_method='debit_card'))
    values,checks=customer_features(events,cutoff,[100, .5, .7, 25])
    assert len(values)==len(ALL) and np.isfinite(values).all() and values[0]==2 and values[1]==8
    # Cancelled baskets cannot affect price/covariance features or spending.
    cancelled=[dict(events[0],order_id='C',order_item_id='C_I',order_status='cancelled',item_total='99999',order_total='99999')]
    other,_=customer_features(events+cancelled,cutoff,[100,.5,.7,25]);assert np.array_equal(values,other)
    unavailable,_=customer_features([],cutoff,[100,np.nan,np.nan,np.nan]);assert np.array_equal(unavailable[:len(SUPPORT)],np.zeros(len(SUPPORT))) and np.isnan(unavailable[len(SUPPORT):]).all()
    future=[dict(events[0],transaction_datetime=(cutoff+timedelta(seconds=1)).isoformat())]
    try:customer_features(future,cutoff,[100,.5,.7,25])
    except AssertionError:pass
    else:raise AssertionError('Future events must fail')
    (OUT/'reports').mkdir(parents=True,exist_ok=True)
    (OUT/'reports/FEATURE_PREFLIGHT.json').write_text(json.dumps(dict(status='PASS',synthetic_only=True,cancellation_invariance=True,empty_history_explicit=True,future_event_rejected=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print('ITEM_PRICE_FEATURE_PREFLIGHT_PASS',flush=True)


def build(raw):
    started=time.time();preflight();dest=OUT/'data/item_price_resilience.parquet';assert not dest.exists(),'Preserve completed feature cache'
    meta=pd.read_parquet(V/'audit_noise/data/application_metadata.parquet',columns=['application_id','customer_id','decision_datetime'])
    assert meta.customer_id.is_unique and meta.application_id.is_unique
    ids={c:i for i,c in enumerate(meta.customer_id)}
    cutoff=[datetime.fromisoformat(x) for x in meta.decision_datetime]
    basecols=['application_id','proposed_monthly_installment','installment_observed_income_ratio','v2_existing_total_burden_observed','v2_account_3m_average_balance_mean']
    # Intentionally exclude default targets even when reading cached matrices.
    frame=pd.concat([pd.read_parquet(ROOT/'data'/f'{part}_features.parquet',columns=basecols) for part in ['train','test']]).set_index('application_id').reindex(meta.application_id)
    financial=frame[basecols[1:]].to_numpy(float)
    values=np.full((len(meta),len(ALL)),np.nan);seen=set();totals=dict(rows=0,future_events=0,nonpositive_price_rows=0,cancelled_items=0,completed_orders180=0)
    with Path(raw).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for customer,events in groupby(reader,key=lambda r:r['customer_id']):
            assert customer in ids and customer not in seen,'Customer FK or block continuity failure'
            seen.add(customer);i=ids[customer]
            row,checks=customer_features(events,cutoff[i],financial[i]);values[i]=row
            for name in totals:totals[name]+=checks[name]
            if len(seen)%10000==0:print('STREAMED_CUSTOMERS',len(seen),'items',totals['rows'],'seconds',round(time.time()-started,1),flush=True)
    absent=set(ids)-seen
    for customer in absent:
        i=ids[customer];values[i]=customer_features([],cutoff[i],financial[i])[0]
    result=pd.DataFrame(values,columns=ALL);result.insert(0,'application_id',meta.application_id.to_numpy());assert result.application_id.is_unique
    dest.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(dest,index=False)
    pd.DataFrame(catalog()).to_csv(OUT/'reports/FEATURE_CATALOG.csv',index=False)
    with Path(raw).open('rb') as f:raw_sha=hashlib.file_digest(f,'sha256').hexdigest()
    with dest.open('rb') as f:feature_sha=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(status='COMPLETE',rows=len(result),features=ALL,groups=dict(support=SUPPORT,price_change=PRICE,order_coherence=COHERENCE,cross_capacity=CROSS),
                raw_ecommerce_sha256=raw_sha,feature_sha256=feature_sha,checks=totals,customers_with_items=len(seen),customers_without_items=len(absent),customer_block_continuity_verified=True,
                target_used=False,population_reference_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                finite_feature_counts={c:int(result[c].notna().sum()) for c in ALL},elapsed_seconds=time.time()-started)
    (OUT/'reports/FEATURE_BUILD.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--raw',default=str(Path.home()/'Downloads/data_logika/ecommerce_transactions.csv'));a=p.parse_args()
    if a.preflight:preflight()
    else:build(a.raw)
