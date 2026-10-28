"""Fresh Kumo inference using a supplied, locally verified checkpoint."""
import importlib
import os,gc,hashlib,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

class HashedWriter:
    """Hash the exact sequential bytes written by torch's ZIP serializer."""
    def __init__(self,file):self.file=file;self.digest=hashlib.sha256();self.bytes=0
    def write(self,data):
        written=self.file.write(data)
        assert written==len(data),'Short checkpoint write'
        self.digest.update(data);self.bytes+=written;return written
    def flush(self):return self.file.flush()
    def tell(self):return self.file.tell()

class Engine:
    def __init__(self,checkpoint,columns,context_data):
        os.environ['SDM_CHUNK_MEMORY_FRACTION']='0.02'
        import torch,sdm
        self.torch=torch;self.sdm=sdm;self.columns=columns
        torch.set_num_threads(2)
        assert torch.cuda.is_available(),'Fresh Kumo inference requires a compatible CUDA device'
        torch.cuda.set_per_process_memory_fraction(.90)
        frame=context_data if isinstance(context_data,pd.DataFrame) else pd.read_parquet(context_data);self.x=self.table(frame)
        self.y=sdm.TableTensor.from_pandas(pd.DataFrame({'target':frame.target.astype(int)}),
            stypes={'target':'categorical'},device='cuda')
        module=importlib.import_module(sdm.models.KumoTabular.__module__)
        def resolve(**kw):
            assert kw['repo_id']=='nvidia/Kumo-Tabular' and kw['filename']=='large/classifier.pt' and kw['revision']=='v1.0.1'
            return str(Path(checkpoint).resolve())
        with patch.object(module,'download_checkpoint',resolve):
            self.model=sdm.models.KumoTabular(task='classification',size='large',device='cuda')
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)

    def table(self,frame):
        x=frame[self.columns].copy();types={}
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                x[col]=x[col].replace([np.inf,-np.inf],np.nan);types[col]='numerical'
            else:x[col]=x[col].astype('string');types[col]='categorical'
        return self.sdm.TableTensor.from_pandas(x,stypes=types,device='cuda')

    def predict_fold(self,fold,indices,query):
        return self.fit_predict(indices,query,2026+fold)

    def fit_predict(self,indices,query,seed,cache_path=None):
        t=self.torch;xt=self.table(query);out=[]
        with t.inference_mode(),t.amp.autocast('cuda',dtype=t.float16):
            self.model.fit(self.x[indices],self.y[indices],num_estimators=8,estimator_batch_size=1,
                generator=t.Generator(device='cuda').manual_seed(seed))
            for start in range(0,len(xt),512):
                out.append(self.model.predict(xt[start:start+512])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
            repeated=self.model.predict(xt[:512])[:,'1'].numerical.reshape(-1).float().cpu().numpy()
            self.last_prediction_replay_error=float(np.max(np.abs(repeated-out[0])))
        if cache_path is not None:
            cache_path=Path(cache_path);temporary=cache_path.with_suffix('.partial.pt')
            assert not cache_path.exists(), 'Preserve completed fitted context'
            started=time.time()
            with temporary.open('wb') as file:
                writer=HashedWriter(file);t.save(self.model._cache,writer)
                writer.flush();os.fdatasync(file.fileno())
                self.last_cache=dict(sha256=writer.digest.hexdigest(),bytes=writer.bytes,
                                     write_seconds=time.time()-started)
                # Advise only this owned, completed file; never drop host caches.
                if hasattr(os,'posix_fadvise'):
                    os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            assert temporary.stat().st_size==self.last_cache['bytes']
            temporary.replace(cache_path)
        self.model.clear();del xt;gc.collect();t.cuda.empty_cache()
        return np.concatenate(out)

    def replay(self,cache_path,query):
        t=self.torch;xt=self.table(query);out=[]
        self.model._cache=t.load(cache_path,weights_only=False)
        with t.inference_mode(),t.amp.autocast('cuda',dtype=t.float16):
            for start in range(0,len(xt),512):
                out.append(self.model.predict(xt[start:start+512])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
        self.model.clear();del xt;gc.collect();t.cuda.empty_cache()
        return np.concatenate(out)
