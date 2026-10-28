import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch
from metaexp.common import read,write,sha,code_hash
from metaexp.coordinator import Transport
import scheduler_parallel as parallel
import rpc_parallel
import node_executor

class SchedulerTests(unittest.TestCase):
    def config(self,root=None):
        now=time.time()
        return {'id':'scheduler_test','started':now,'deadline':now+43200,'code_hash':'fixture','machines':{
            'pc1':{'workers':16,'opencl':True,'root':str(root or '/fixture')},'pc2':{'workers':7,'root':str(root or '/fixture')},'pc3':{'workers':7,'root':str(root or '/fixture')}}}
    def test_batch_delivery_uses_one_call_per_host(self):
        with tempfile.TemporaryDirectory() as directory:
            tx=parallel.BatchTransport(Path(directory),self.config());calls=[]
            for host in parallel.HOSTS:
                for i in range(40):tx.call(host,{'action':'push','tasks':[{'id':host+str(i),'recipe':{'id':str(i)}}]})
            def send(_self,host,request,timeout=30):calls.append((host,len(request['tasks'])));return {'accepted':len(request['tasks'])}
            with patch.object(Transport,'call',send):tx.flush();tx.flush()
            self.assertCountEqual(calls,[('pc1',40),('pc2',40),('pc3',40)])
    def test_failed_delivery_retries_same_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            tx=parallel.BatchTransport(Path(directory),self.config());task={'id':'once','recipe':{'id':'once'}}
            tx.call('pc1',{'action':'push','tasks':[task]})
            with patch.object(Transport,'call',side_effect=RuntimeError('SSH disconnected')):tx.flush()
            self.assertEqual(tx.buffer['pc1']['once'],task)
            with patch.object(Transport,'call',return_value={'accepted':1}) as sending:tx.flush();tx.call('pc1',{'action':'push','tasks':[task]});tx.flush()
            self.assertEqual(sending.call_count,1)
    def test_changed_buffered_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tx=parallel.BatchTransport(Path(directory),self.config());tx.call('pc1',{'action':'push','tasks':[{'id':'one','until':1}]})
            with self.assertRaises(ValueError):tx.call('pc1',{'action':'push','tasks':[{'id':'one','until':2}]})
    def test_gpu_priority_changes_queue_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cpu={'id':'cpu','device':'cpu'};gpu={'id':'gpu','device':'cuda'}
            write(root/'inbox/cpu.json',cpu);write(root/'inbox/gpu.json',gpu);before=sha(root/'inbox/gpu.json')
            rpc_parallel.prioritize_gpu(root,[cpu,gpu])
            self.assertLess((root/'inbox/gpu.json').stat().st_mtime,(root/'inbox/cpu.json').stat().st_mtime)
            self.assertEqual(sha(root/'inbox/gpu.json'),before)
    def test_enqueue_phase_flushes_learned_ensemble_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            tx=parallel.BatchTransport(Path(directory),self.config());tx.buffering=False
            with patch.object(Transport,'call',return_value={'accepted':1}) as sending:tx.call('pc3',{'action':'push','tasks':[{'id':'nested'}]})
            self.assertEqual(sending.call_count,1)
    def fixture(self,root,tid):
        folder=root/'runs'/tid;folder.mkdir(parents=True);write(folder/'status.json',{'state':'completed'});write(folder/'identity.json',{'task':tid})
        (folder/'prediction.npy').write_bytes(b'prediction-evidence');(folder/'worker.log').write_bytes(b'log')
        write(folder/'complete.json',{'files':{'prediction.npy':sha(folder/'prediction.npy')}});return folder
    def test_batch_archive_round_trip_and_path_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for tid in ['one','two']:self.fixture(root,tid)
            stream=io.BytesIO();rpc_parallel.pack_many(root,['one','two'],stream)
            with tarfile.open(fileobj=io.BytesIO(stream.getvalue()),mode='r:gz') as archive:
                self.assertEqual(archive.extractfile('runs/two/prediction.npy').read(),b'prediction-evidence')
            with self.assertRaises(ValueError):rpc_parallel.pack_many(root,['../secret'],io.BytesIO())
    def test_compaction_requires_exact_verified_archive_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=self.fixture(root,'one')
            with self.assertRaises(ValueError):rpc_parallel.compact(root,[{'id':'one','complete_sha256':'wrong'}])
            self.assertTrue((folder/'prediction.npy').exists())
            receipt={'id':'one','complete_sha256':sha(folder/'complete.json')}
            rpc_parallel.compact(root,[receipt]);rpc_parallel.compact(root,[receipt])
            self.assertTrue((folder/'compacted.json').exists());self.assertTrue((folder/'identity.json').exists());self.assertEqual((folder/'prediction.npy').stat().st_size,0)
    def test_parallel_dispatch_preserves_limits_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config(root);write(root/'campaign.json',cfg)
            (root/'rpc_parallel.py').symlink_to(Path(rpc_parallel.__file__).resolve())
            (root/'node_executor.py').symlink_to(Path(node_executor.__file__).resolve())
            recipes=[{'id':'r_'+str(i),'model':['lgb','cat','xgb','linear'][i%4],'representation':{'id':'f_'+str(i)}} for i in range(400)]
            write(root/'catalogue.json',{'recipes':recipes,'control':'unused'})
            write(root/'scheduler_revision.json',{'script_sha256':sha(Path(parallel.__file__)),'rpc_sha256':sha(Path(rpc_parallel.__file__)),'training_code_hash':'fixture','started':cfg['started'],'deadline':cfg['deadline']})
            with patch.object(parallel,'code_hash',return_value='fixture'):coordinator=parallel.ParallelCoordinator(root)
            with patch.object(Transport,'call',return_value={'accepted':1}):
                coordinator.dispatch(time.time()+3600);first=coordinator.ledger.all();coordinator.dispatch(time.time()+3600);second=coordinator.ledger.all()
            self.assertEqual(len(first),len(second));self.assertGreater(len(first),30)
            for record in second:
                task=json.loads(record['task']);self.assertEqual(task['recipe'],recipes[int(task['recipe']['id'][2:])]);self.assertEqual(task['scheduler_revision'],coordinator.revision)
                if record['host']=='pc1':self.assertNotEqual(task['device'],'cuda')
            self.assertEqual(read(root/'campaign.json'),cfg)
            self.assertEqual(len(first),len(recipes));self.assertEqual(coordinator.cfg['machines']['pc2']['workers'],7)
            self.assertEqual(coordinator.cfg['machines']['pc1']['workers'],16)
            self.assertTrue((root/'local_catalogue_shard.json').exists())
    def test_overlay_keeps_frozen_training_source_identity(self):
        root=Path(__file__).resolve().parent
        if not (root/'campaign.json').exists():root=root.parent
        cfg=read(root/'campaign.json')
        self.assertEqual(code_hash(root),cfg['code_hash'])
    def test_cpu_capacity_runs_remaining_cat_and_xgb_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config(root);write(root/'campaign.json',cfg)
            (root/'rpc_parallel.py').symlink_to(Path(rpc_parallel.__file__).resolve())
            (root/'node_executor.py').symlink_to(Path(node_executor.__file__).resolve())
            recipes=[{'id':'r_'+str(i),'model':['cat','xgb'][i%2],'representation':{'id':'f_'+str(i)}} for i in range(300)]
            write(root/'catalogue.json',{'recipes':recipes,'control':'unused'})
            write(root/'scheduler_revision.json',{'script_sha256':sha(Path(parallel.__file__)),'rpc_sha256':sha(Path(rpc_parallel.__file__)),'training_code_hash':'fixture','started':cfg['started'],'deadline':cfg['deadline']})
            with patch.object(parallel,'code_hash',return_value='fixture'):coordinator=parallel.ParallelCoordinator(root)
            with patch.object(Transport,'call',return_value={'accepted':1}):coordinator.dispatch(time.time()+3600)
            rows=coordinator.ledger.all();tasks=[(r['host'],json.loads(r['task'])) for r in rows]
            for host in parallel.HOSTS:self.assertTrue(any(h==host and t['device']=='cpu' for h,t in tasks))
            for host in ['pc2','pc3']:self.assertTrue(any(h==host and t['device']=='cuda' for h,t in tasks))
            self.assertEqual(len({r['id'] for r in rows}),len(rows));self.assertEqual(read(root/'campaign.json'),cfg)
    def test_confirmation_and_sensitivity_preserve_parent_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config(root);write(root/'campaign.json',cfg)
            (root/'rpc_parallel.py').symlink_to(Path(rpc_parallel.__file__).resolve())
            (root/'node_executor.py').symlink_to(Path(node_executor.__file__).resolve())
            recipe={'id':'cat_recipe','model':'cat','representation':{'id':'features'}}
            write(root/'catalogue.json',{'recipes':[recipe],'control':'unused'});write(root/'runs/parent/scores.json',{'device':'cpu'})
            write(root/'scheduler_revision.json',{'script_sha256':sha(Path(parallel.__file__)),'rpc_sha256':sha(Path(rpc_parallel.__file__)),'training_code_hash':'fixture','started':cfg['started'],'deadline':cfg['deadline']})
            with patch.object(parallel,'code_hash',return_value='fixture'):coordinator=parallel.ParallelCoordinator(root)
            for mode in ['confirmation','repeat','temporal']:
                coordinator.submit('pc2',recipe,time.time()+600,mode=mode,tid=mode,parent_id='parent')
            for record in coordinator.ledger.all():self.assertEqual(json.loads(record['task'])['device'],'cpu')
    def test_local_timeout_starts_at_admission_and_preserves_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'inbox/job.json';write(path,{'until':10000,'execution_timeout_seconds':900});before=sha(path)
            with patch.object(node_executor.time,'time',return_value=1000):value=node_executor.queue_read(path)
            self.assertEqual(value['until'],1900);self.assertEqual(read(path)['until'],10000);self.assertEqual(sha(path),before)
            with patch.object(node_executor.time,'time',return_value=9900):self.assertEqual(node_executor.queue_read(path)['until'],10000)
    def test_shard_install_is_idempotent_and_cannot_change_owner_or_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config(root);write(root/'campaign.json',cfg);recipe={'id':'recipe','model':'cat'};write(root/'catalogue.json',{'recipes':[recipe]})
            task={'id':'job','recipe':recipe,'mode':'cv','device':'cpu','until':cfg['deadline'],'execution_timeout_seconds':900}
            shard={'host':'pc2','campaign':cfg['id'],'started':cfg['started'],'deadline':cfg['deadline'],'training_code_hash':cfg['code_hash'],'catalogue_sha256':sha(root/'catalogue.json'),'tasks':[task]}
            rpc_parallel.install_shard(root,shard);rpc_parallel.install_shard(root,shard)
            self.assertEqual(read(root/'inbox/job.json'),task)
            changed=copy.deepcopy(shard);changed['deadline']+=1
            with self.assertRaises(ValueError):rpc_parallel.install_shard(root,changed)
            cfg['machines']['pc1']['root']='/another/node';write(root/'campaign.json',cfg);changed=copy.deepcopy(shard);changed['host']='pc1'
            with self.assertRaises(ValueError):rpc_parallel.install_shard(root,changed)

if __name__=='__main__':unittest.main(verbosity=2)
