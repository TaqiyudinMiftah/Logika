"""Fresh Kumo inference using a supplied, locally verified checkpoint."""
import importlib
import os,gc
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

class Engine:
    def __init__(self,checkpoint,columns,context_data):
        os.environ['SDM_CHUNK_MEMORY_FRACTION']='0.02'
        import torch,sdm
        self.torch=torch;self.sdm=sdm;self.columns=columns
        torch.set_num_threads(2)
        assert torch.cuda.is_available(),'Fresh Kumo inference requires a compatible CUDA device'
        torch.cuda.set_per_process_memory_fraction(.90)
        frame=pd.read_parquet(context_data);self.x=self.table(frame)
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

    def fit_predict(self,indices,query,seed):
        t=self.torch;xt=self.table(query);out=[]
        with t.inference_mode(),t.amp.autocast('cuda',dtype=t.float16):
            self.model.fit(self.x[indices],self.y[indices],num_estimators=8,estimator_batch_size=1,
                generator=t.Generator(device='cuda').manual_seed(seed))
            for start in range(0,len(xt),512):
                out.append(self.model.predict(xt[start:start+512])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
        self.model.clear();del xt;gc.collect();t.cuda.empty_cache()
        return np.concatenate(out)
