from __future__ import annotations
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from metaexp.common import metrics,write,read,inside,identity
from metaexp.reporting import fast_ks
from metaexp.design import DEFAULT,design

class FakeBank:
    def __init__(self,root=None):
        self.root=Path(root or '.');self.dev=np.arange(300);self.y=np.arange(300)%2;self.folds=np.arange(300)%3
        self.metadata=pd.DataFrame({'application_id':['id'+str(i) for i in self.dev],'decision_datetime':pd.date_range('2020-01-01',periods=300)})
        self.categorical=pd.DataFrame({'employment_type':['e'+str(i) for i in self.dev]})
        self.manifest={'bank_id':'test','info':{'a':{'sources':['application']},'b':{'sources':['profile']}}}
    def columns(self,r):return ['a','b'],['employment_type']
    def cohort(self,fit,rows,names,c):
        rows=np.asarray(rows);return np.column_stack([rows.astype(float),np.where(rows%7==0,np.nan,np.sin(rows))]).astype(np.float32),names

class Acceptance(unittest.TestCase):
    def test_ks_with_ties_and_orientation(self):
        from scipy.stats import ks_2samp
        y=np.array([0,1,0,1,1,0]);p=np.array([.5,.5,.1,.9,.9,.5])
        expected=ks_2samp(p[y==1],p[y==0]).statistic
        self.assertAlmostEqual(metrics(y,p)['ks'],expected);self.assertAlmostEqual(fast_ks(y,p),expected)
    def test_incomplete_oof_rejected(self):
        from metaexp.common import verify_oof
        with self.assertRaises(ValueError):verify_oof(np.array([0,1]),np.array([.1,np.nan]),np.array([0,1]))
    def test_design_deterministic_and_unique(self):
        with tempfile.TemporaryDirectory() as d:
            a=design(Path(d),64);b=design(Path(d),64)
            self.assertEqual(a,b);self.assertEqual(len(a),256);self.assertEqual(len({r['id'] for r in a}),256)
            self.assertEqual({r['representation']['family'] for r in a},set(['sources','repayment','bureau','account','shopping','cross_source','preprocessing','reduction']))
    def test_full_catalogue_has_1024_effective_representations(self):
        with tempfile.TemporaryDirectory() as d:
            rows=design(Path(d),1024)
            self.assertEqual(len(rows),4096)
            self.assertEqual(len({r['representation']['id'] for r in rows}),1024)
    def test_preprocessor_uses_training_rows(self):
        from metaexp.pipeline import Preprocessor
        bank=FakeBank();rep={'config':copy.deepcopy(DEFAULT),'family':'sources'};fit=np.arange(120);query=np.arange(120,150)
        pre=Preprocessor(bank,rep,'linear');x=pre.fit_transform(fit,bank.y[fit]);before=pre.fill.copy();q=pre.transform(query)
        self.assertEqual(pre.audit['fit_rows_hash'],identity(fit.tolist()));np.testing.assert_array_equal(pre.fill,before)
        self.assertEqual(x.shape[1],q.shape[1]);self.assertFalse(np.intersect1d(pre.fit_rows,query).size)
    def test_target_encoding_is_inner_crossfit(self):
        from metaexp.pipeline import Preprocessor
        bank=FakeBank();cfg=copy.deepcopy(DEFAULT);cfg['encoding']='target';rep={'config':cfg,'family':'preprocessing'}
        pre=Preprocessor(bank,rep,'lgb');x=pre.fit_transform(np.arange(180),bank.y[:180])
        self.assertEqual(pre.audit['target_encoding_inner_folds'],3)
        self.assertTrue(np.all((x[:,-1]>.4)&(x[:,-1]<.6)))
    def test_queue_duplicate_delivery_and_changed_recipe(self):
        from metaexp.runtime import enqueue
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);recipe={'id':'p_one','model':'lgb'};write(root/'catalogue.json',{'recipes':[recipe]})
            task={'id':'p_one','recipe':recipe,'mode':'cv','until':100}
            self.assertEqual(enqueue(root,[task]),1);self.assertEqual(enqueue(root,[task]),0)
            changed=copy.deepcopy(task);changed['until']=200
            with self.assertRaises(ValueError):enqueue(root,[changed])
            self.assertEqual(read(root/'inbox/p_one.json')['until'],100)
    def test_one_assignment_per_id_during_ssh_loss(self):
        from metaexp.coordinator import Ledger
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'db');task={'id':'p_one','recipe':{'id':'p_one'}};ledger.insert(task,'pc1')
            with self.assertRaises(sqlite3.IntegrityError):ledger.insert(task,'pc3')
            self.assertEqual(ledger.all()[0]['host'],'pc1');self.assertEqual(ledger.all()[0]['state'],'assigned')
    def test_expired_campaign_cannot_reset_deadline(self):
        from metaexp.coordinator import launch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'campaign.json';write(p,{'deadline':1,'started':0});before=p.read_bytes()
            with self.assertRaises(ValueError):launch(Path(d))
            self.assertEqual(before,p.read_bytes())
    def test_unsafe_result_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):inside(d,'../secret')
            with self.assertRaises(ValueError):inside(d,'/etc/passwd')
    def test_preflight_collection_uses_authoritative_host_proof(self):
        from metaexp.deployment import publish_preflights
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for host in ['pc1','pc2','pc3']:
                for name in ['pc1','pc2','pc3']:write(root/'preflight_hosts'/host/(name+'.json'),{'host':name,'source':'fresh' if name==host else 'stale'})
            publish_preflights(root)
            for host in ['pc1','pc2','pc3']:self.assertEqual(read(root/'preflight'/(host+'.json'))['source'],'fresh')
    def test_rpc_bank_export_roundtrip(self):
        import io,tarfile
        from metaexp.runtime import rpc
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'bank').mkdir();(root/'bank/payload').write_bytes(b'feature-bank-test')
            buffer=io.BytesIO()
            with patch('metaexp.runtime.sys.stdin',io.StringIO('{"action":"bank_export"}')),patch('metaexp.runtime.sys.stdout',types.SimpleNamespace(buffer=buffer)):
                rpc(root)
            with tarfile.open(fileobj=io.BytesIO(buffer.getvalue()),mode='r:gz') as tar:
                self.assertEqual(tar.extractfile('bank/payload').read(),b'feature-bank-test')
    def test_crashed_fold_never_completes_recipe(self):
        from metaexp.worker import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bank=FakeBank(root);rep={'id':'f_test','family':'sources','config':copy.deepcopy(DEFAULT)}
            task={'id':'p_crash','recipe':{'id':'p_crash','model':'lgb','representation':rep,'seed':2026},'mode':'cv'}
            calls=[]
            def fake(bank,recipe,fitrows,y,queryrows,*args):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('intentional worker crash')
                pre=types.SimpleNamespace(audit={'fit_rows_hash':identity(fitrows.tolist())})
                return np.linspace(.1,.9,len(queryrows)),pre,object(),1,np.linspace(.1,.9,257),[]
            with patch('metaexp.worker.Bank',return_value=bank),patch('metaexp.worker.fit_predict',side_effect=fake):
                with self.assertRaisesRegex(RuntimeError,'intentional'):execute(root,task)
            self.assertFalse((root/'runs/p_crash/complete.json').exists());self.assertEqual(read(root/'runs/p_crash/status.json')['state'],'failed')
            self.assertTrue((root/'runs/p_crash/fold_0/complete.json').exists())
    def test_confirmation_requires_frozen_parent(self):
        from metaexp.worker import sensitivity
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'finalists.json',{'ids':['other']});bank=FakeBank(root)
            with patch('metaexp.worker.fit_predict') as fitting:
                with self.assertRaises(ValueError):sensitivity(root,bank,{'recipe':{},'mode':'confirmation','parent_id':'unselected'},root)
                fitting.assert_not_called()
    def test_nested_fit_excludes_outer_validation(self):
        from metaexp.ensembles import nested
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bank=FakeBank(root);calls=[]
            def fake(bank,recipe,fitrows,y,queryrows,*args):
                fold=len(calls)//8;outer_training=set(bank.dev[bank.folds!=fold]);calls.append(1)
                self.assertTrue(set(fitrows)<=outer_training);self.assertFalse(set(fitrows)&set(queryrows))
                np.testing.assert_array_equal(y,bank.y[np.asarray(fitrows)])
                return np.linspace(.2,.8,len(queryrows)),None,None,None,None,None
            task={'recipe':{'bases':[{'model':'lgb'},{'model':'linear'}],'method':'weights'},'device':'cpu'}
            with patch('metaexp.ensembles.fit_predict',side_effect=fake):result=nested(root,bank,task,root)
            self.assertEqual(len(calls),24);self.assertEqual(result['folds'],3)
    def test_real_daemon_deadline_and_consolidated_report(self):
        import signal,sys,time,threading,os
        import metaexp
        from metaexp.features import Bank
        from metaexp.runtime import enqueue,daemon
        from metaexp.reporting import report,verify_run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bank=root/'bank';bank.mkdir()
            (root/'metaexp').symlink_to(Path(metaexp.__file__).parent,target_is_directory=True)
            (root/'.venv').symlink_to(Path(sys.executable).parents[1],target_is_directory=True)
            rng=np.random.default_rng(1);matrix=rng.normal(size=(400,3)).astype(np.float32)
            np.save(bank/'numeric.npy',matrix)
            pd.DataFrame({'employment_type':['salaried']*400}).to_parquet(bank/'categorical.parquet',index=False)
            pd.DataFrame({'application_id':['id'+str(i) for i in range(400)],'customer_id':['c'+str(i) for i in range(400)],'decision_datetime':pd.date_range('2020-01-01',periods=400),'split':['train']*300+['test']*100}).to_parquet(bank/'metadata.parquet',index=False)
            for name,value in [('development_rows',np.arange(240)),('confirmation_rows',np.arange(240,300)),('development_y',np.arange(240)%2),('confirmation_y',np.arange(60)%2),('folds',np.arange(240)%3)]:np.save(bank/(name+'.npy'),value)
            names=['loan_amount','age','m_repayment_all_count'];manifest={'bank_id':'synthetic_runtime_test','columns':names,'categorical_columns':['employment_type'],'priors':[],
                'info':{n:{'sources':['application'],'window':'all','tags':['raw']} for n in names},'views':{v:names+['employment_type'] for v in ['v3','base','extended']}}
            write(bank/'manifest.json',manifest);recipes=design(root,8);recipe=recipes[0];now=time.time()
            cfg={'id':'synthetic_runtime_test','started':now,'deadline':now+30,'code_hash':'fixture','machines':{'pc2':{'workers':1,'admission_gib':4,'run_disk_gib':1}}}
            write(root/'campaign.json',cfg);task={'id':recipe['id'],'recipe':recipe,'mode':'cv','device':'cpu','until':now+29}
            enqueue(root,[task]);oldterm=signal.getsignal(signal.SIGTERM);oldint=signal.getsignal(signal.SIGINT)
            def stop_after_completion():
                for _ in range(200):
                    p=root/'runs'/recipe['id']/'status.json'
                    if p.exists() and read(p)['state'] in ['completed','failed']:break
                    time.sleep(.05)
                os.kill(os.getpid(),signal.SIGTERM)
            watcher=threading.Thread(target=stop_after_completion)
            try:
                watcher.start();daemon(root,'pc2');watcher.join()
            finally:signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGINT,oldint)
            self.assertTrue((root/'daemon_stopped.json').exists())
            complete=root/'runs'/recipe['id']/'complete.json'
            if not complete.exists():self.fail((root/'runs'/recipe['id']/'worker.log').read_text())
            verify_run(root,recipe['id'],Bank(root))
            summary=report(root,final=True);self.assertEqual(summary['completed_distinct_pipelines'],1)
            self.assertTrue((root/'reports/final_report.md').is_file())
            before_heartbeat=(root/'heartbeat.json').read_bytes()
            cfg['deadline']=time.time()-1;write(root/'campaign.json',cfg)
            try:daemon(root,'pc2')
            finally:signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGINT,oldint)
            self.assertEqual((root/'heartbeat.json').read_bytes(),before_heartbeat)

if __name__=='__main__':unittest.main()
