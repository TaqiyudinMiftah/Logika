"""A real training-only gradient check; never evaluates outer validation labels."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
os.environ['KUMO_MEMORY_FRACTION']='.30'
from pathlib import Path
import json,time,hashlib,subprocess
import numpy as np
import pandas as pd
from gpu_backend import Engine
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pretrained_hash(model,selected):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if name in selected:continue
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().view(-1).view(dtype=__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()
def choose_parameters(model):
    selected={}
    for name,p in model.named_parameters():
        p.requires_grad_(False)
        if name.startswith('models.classification.icl_block.layers.') and name.endswith('.attn.sdpa.query_scaling.head_scale'):
            p.requires_grad_(True);selected[name]=p
    return selected
def main():
    start=time.time();plan=json.loads((ROOT/'plan.json').read_text())
    assert time.time()<plan['original_deadline']
    assert not (ROOT/'full_shape_preflight.json').exists(),'Preserve existing gradient evidence'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    assert free>=plan['minimum_free_VRAM_MiB'],f'Not enough safe memory headroom:{free}MiB'
    assert sha(plan['checkpoint'])==plan['checkpoint_sha256']
    data=pd.read_parquet(ROOT/'data/context0.parquet');assert len(data)==50000 and data.application_id.is_unique
    columns=json.loads((ROOT/'data/columns.json').read_text())
    assert len(columns)==96 and not set(columns)&{'target','application_id','customer_id','split'}
    rng=np.random.default_rng(20261108);sample=rng.choice(len(data),512,replace=False)
    support,query=sample[:384],sample[384:];assert not set(support)&set(query)
    assert set(data.target.iloc[support].astype(int))=={0,1}
    engine=Engine(plan['checkpoint'],columns,data);t=engine.torch;model=engine.model
    xq=engine.table(data.iloc[query].drop(columns='target'))
    def predict(seed=2026,training=False):
        model.train(training)
        with t.amp.autocast('cuda',dtype=t.float16):
            out=model(engine.x[support],engine.y[support],xq,num_estimators=1,
                estimator_batch_size=1,generator=t.Generator(device='cuda').manual_seed(seed))
            return out[:,'1'].numerical.reshape(-1).float()
    with t.inference_mode():before=predict().cpu().numpy().copy()
    selected=choose_parameters(model);count=sum(p.numel()for p in selected.values())
    assert count==plan['expected_trainable_parameters'],count
    original={name:p.detach().clone()for name,p in selected.items()}
    frozen_before=pretrained_hash(model,selected)
    with t.inference_mode():zero=predict().cpu().numpy().copy()
    zero_error=float(np.max(abs(before-zero)));assert zero_error<2e-5,zero_error
    optimizer=t.optim.AdamW(list(selected.values()),lr=.001,weight_decay=0)
    prediction=predict(training=True);assert prediction.requires_grad
    y=t.tensor(data.target.iloc[query].to_numpy(dtype=np.float32),device='cuda')
    loss=t.nn.functional.binary_cross_entropy(prediction.clamp(1e-6,1-1e-6),y)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    gradients={name:p.grad.detach().float().cpu()for name,p in selected.items()if p.grad is not None}
    assert gradients and all(t.isfinite(g).all()for g in gradients.values())
    norm=float(sum(float(g.square().sum())for g in gradients.values())**.5);assert norm>0
    assert all(p.grad is None for name,p in model.named_parameters()if name not in selected)
    optimizer.step()
    with t.no_grad():
        for name,p in selected.items():p.copy_(original[name]+(p-original[name]).clamp(-.25,.25))
    max_change=max(float((p-original[name]).abs().max())for name,p in selected.items());assert max_change>0
    frozen_after=pretrained_hash(model,selected);assert frozen_after==frozen_before
    model.eval()
    with t.inference_mode():after=predict().cpu().numpy().copy()
    (ROOT/'artifacts').mkdir(exist_ok=True)
    t.save(dict(selected_before={n:p.cpu()for n,p in original.items()},
        selected_after={n:p.detach().cpu()for n,p in selected.items()},gradients=gradients,
        optimizer=optimizer.state_dict()),ROOT/'artifacts/full_shape_step.pt')
    np.savez_compressed(ROOT/'artifacts/full_shape_predictions.npz',before=before,zero=zero,after=after,
        support_indices=support,query_indices=query)
    total=sum(p.numel()for p in model.parameters())
    result=dict(status='passed',scope='Gradient engineering preflight only; no OOF performance estimate.',
        checkpoint_sha256=plan['checkpoint_sha256'],trainable_parameters=count,total_parameters=total,
        trainable_fraction=count/total,selected_parameter_names=sorted(selected),
        frozen_nonselected_parameters_sha256=frozen_after,zero_update_max_error=zero_error,
        gradient_norm=norm,selected_max_update=max_change,preflight_training_loss=float(loss.detach()),
        prediction_change=float(np.max(abs(after-before))),context_source_rows=50000,
        support_rows=384,query_rows=128,both_splits_from_training_context_only=True,
        outer_validation_or_correction_labels_staged=False,peak_vram_bytes=t.cuda.max_memory_allocated(),
        device=t.cuda.get_device_name(),seconds=time.time()-start,
        model_training_admitted=False)
    (ROOT/'full_shape_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
