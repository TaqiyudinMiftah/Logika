"""Generic Kumo-Tabular packet runner (node2001 GPU1 validation and Colab A100 production).

Packet layout (one directory per fit):
  X.parquet       rows = [context rows..., query rows...]; columns = exact CV view, CV order
  y_context.npy   int labels for the first n_context rows
  query_rows.npy  int64 global row ids (FULL frame 0..99999, or CV 0..69999) for the query rows
  meta.json       {"model","fold","part","n_context","seed","num_estimators","size"} (+ free-form notes)
Output: <out>/<relative packet path>/pred.npz with query_rows, raw (float32) + run.json.

Semantics are identical to kumo_fit_predict in build_unified_tables_and_run_kumo.py:
numeric columns -> numerical; others -> categorical with vocabulary restricted to context rows;
fp16 autocast; estimator_batch_size=1; generator seeded on device; predict P(class "1").
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("SDM_CHUNK_MEMORY_FRACTION", "0.02")
import argparse, gc, hashlib, importlib, json, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

LARGE_SHA = "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def make_kumo(ckpt, size, device):
    import sdm
    module = importlib.import_module(sdm.models.KumoTabular.__module__)
    with patch.object(module, "download_checkpoint", lambda **kw: str(ckpt)):
        return sdm.models.KumoTabular(task="classification", size=size, device=device)


def run_packet(pdir, ckpt, device, pred_batch, mem_fraction):
    import torch, sdm
    torch.set_grad_enabled(False)
    if mem_fraction:
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
    meta = json.loads((pdir / "meta.json").read_text())
    X = pd.read_parquet(pdir / "X.parquet")
    y = np.load(pdir / "y_context.npy").astype(int)
    q = np.load(pdir / "query_rows.npy").astype(np.int64)
    n = int(meta["n_context"])
    assert len(y) == n and len(X) == n + len(q), (len(y), n, len(X), len(q))
    assert X.shape[1] <= 99
    ordered = X.reset_index(drop=True).copy()
    types = {}
    for c in ordered.columns:
        if pd.api.types.is_numeric_dtype(ordered[c]):
            types[c] = "numerical"
        else:
            vocab = set(ordered[c].iloc[:n].dropna().astype(str))
            ordered[c] = ordered[c].astype("string").where(ordered[c].astype("string").isin(vocab))
            types[c] = "categorical"
    table = sdm.TableTensor.from_pandas(ordered, stypes=types, device=device)
    target = sdm.TableTensor.from_pandas(pd.DataFrame({"target": y}), stypes={"target": "categorical"}, device=device)
    model = make_kumo(ckpt, meta.get("size", "large"), device)
    out = []
    with torch.inference_mode(), torch.amp.autocast(device, dtype=torch.float16, enabled=True):
        model.fit(table[:n], target, num_estimators=int(meta.get("num_estimators", 8)), estimator_batch_size=1,
                  generator=torch.Generator(device=device).manual_seed(int(meta["seed"])))
        for s in range(n, len(table), pred_batch):
            out.append(model.predict(table[s:s + pred_batch])[:, "1"].numerical.reshape(-1).float().cpu().numpy())
    raw = np.concatenate(out).astype(np.float32)
    model.clear()
    del table, target, model
    gc.collect()
    torch.cuda.empty_cache()
    assert len(raw) == len(q) and np.isfinite(raw).all()
    return q, raw, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pred-batch", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.0)
    ap.add_argument("--only", nargs="*", default=None, help="relative packet paths to run (default: all)")
    args = ap.parse_args()
    root, outroot = Path(args.packets), Path(args.out)
    assert sha(args.ckpt) == LARGE_SHA, "checkpoint hash mismatch"
    pdirs = sorted(p.parent for p in root.rglob("meta.json"))
    if args.only:
        pdirs = [root / r for r in args.only]
    # largest jobs first keeps the GPU saturated and surfaces OOM early
    pdirs.sort(key=lambda p: -json.loads((p / "meta.json").read_text())["n_context"])
    for pdir in pdirs:
        rel = pdir.relative_to(root)
        dest = outroot / rel
        if (dest / "pred.npz").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        t = time.time()
        q, raw, meta = run_packet(pdir, args.ckpt, args.device, args.pred_batch, args.mem_fraction)
        np.savez_compressed(dest / "pred.tmp.npz", query_rows=q, raw=raw)
        os.replace(dest / "pred.tmp.npz", dest / "pred.npz")
        (dest / "run.json").write_text(json.dumps(dict(packet=str(rel), seconds=time.time() - t, n_context=meta["n_context"],
                                                       n_query=int(len(q)), seed=meta["seed"], mean=float(raw.mean()),
                                                       x_sha256=sha(pdir / "X.parquet")), indent=1))
        print(json.dumps(dict(done=str(rel), seconds=round(time.time() - t, 1), mean=float(raw.mean()))), flush=True)


if __name__ == "__main__":
    main()
