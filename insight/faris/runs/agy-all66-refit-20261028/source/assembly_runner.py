"""Reconstruct ALL66 fixed routed mix and refit its spline/head components.

The 0.534013 mixture was selected using reused development OOF outcomes.
This runner makes no independent validation or public-score improvement claim.
No fitted weights or preprocessing objects are serialized.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from threadpoolctl import threadpool_limits
from spline_models import (make_scaled_spline_matrix, fit_logistic_irls, predict_irls,
                           fixed_spline_fit, direct_spline_update)

DEFAULT_HOME = Path("/mnt/home/faris")
RECIPES = {
    "pop_A": ["h29_pop_spec", "raw_fe_cov", "h29_pop_spec"],
    "pop_B": ["per29_pop_spec"],
    "pop_C1": ["per29_champ_s7731", "raw_kumo56", "mat18_kumo56", "raw_tr_mat_orig", "raw_shadow91", "h29_shadow91"],
    "pop_C2": ["mat18_fe_mat", "h29_pop_spec", "sp200_56k", "mat18_fe_cov", "per29_fe_cov"],
}
REQUIRED = set(x for values in RECIPES.values() for x in values)
BACKBONES = ["pop_spec", "fe_cov", "fe_mat", "champ_s7731", "shadow91", "tr_mat_orig", "kumo56"]
SPECTRAL = ["beh_spectral_" + channel + "_" + name
            for channel in ("spend", "orders")
            for name in ("power_7d", "power_14d", "power_30d", "periodicity_strength", "dominant_cycle_days")]
SPECTRAL += ["beh_spectral_window_supported"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def paths(home):
    return home / "logika-faris-exploration", home / "logika-frozen-shadow-20261026", home / "logika-joint-20261028"


def modules(exp):
    source = exp / "transfer-20261025/source"
    sys.path.insert(0, str(source))
    import baseline
    import feature_head
    return baseline, feature_head


def ks(y, p):
    order = np.argsort(p, kind="mergesort"); labels = y[order]; values = p[order]
    end = np.r_[values[:-1] != values[1:], True]
    return float(np.max(np.abs(np.cumsum(labels == 1) / np.sum(labels == 1) - np.cumsum(labels == 0) / np.sum(labels == 0))[end]))


def align_prediction(path, want, *, universe=100000, fallback="raw"):
    with np.load(path, allow_pickle=False) as z:
        key = next((k for k in [fallback, "prediction", "raw", "proba"] if k in z), None)
        assert key is not None, (str(path), z.files)
        values = np.asarray(z[key], dtype=np.float64).reshape(-1)
        row_key = next((k for k in ["rows", "query_rows"] if k in z), None)
        if row_key is None:
            assert len(values) == len(want), (path, len(values), len(want))
            return values
        rows = np.asarray(z[row_key], dtype=np.int64)
    assert len(rows) == len(values) and len(np.unique(rows)) == len(rows)
    if np.array_equal(rows, want): return values
    lookup = np.full(universe, np.nan)
    lookup[rows] = values
    result = lookup[want]
    assert np.isfinite(result).all(), str(path)
    return result


def raw_replay(exp, frozen, joint, fold, roles):
    want = np.r_[roles["correction"], roles["validation"]]
    files = {
        "pop_spec": joint / f"kumo_runs/kumo_pop_specialist/fold{fold}/prediction.npz",
        "fe_cov": exp / f"coverage-information-fe-20261021/runs/fold{fold}/raw.npz",
        "fe_mat": exp / f"mature-mask-preserved-fe-20261021/runs/fold{fold}/raw.npz",
        "champ_s7731": frozen / f"runs/kumo_large_champion90_s7731/fold{fold}/prediction.npz",
        "shadow91": frozen / f"runs/kumo_large_shadow91/fold{fold}/prediction.npz",
        "tr_mat_orig": exp / f"transfer-20261025/core_maturity/fold{fold}/original_raw.npz",
    }
    result = {name: align_prediction(path, want, universe=70000) for name, path in files.items()}
    meta_path = exp / f"nested-correction-20261021/heads/fold{fold}/meta_inputs.npz"
    raw_path = exp / f"nested-correction-20261021/runs/job{fold*6+5}/raw.npz"
    lookup = np.full(70000, np.nan)
    for path in [meta_path, raw_path]:
        with np.load(path, allow_pickle=False) as z: lookup[z["rows"]] = z["prediction"]
    result["kumo56"] = lookup[want]
    files.update(kumo56_inner=meta_path, kumo56_validation=raw_path)
    assert all(np.isfinite(v).all() and len(v) == 20000 for v in result.values())
    return result, files


def candidates(frame, raw, y_correction, head18, spline_cor, spline_mat, spline_per, spline200_query, exp, frame_overrides=None):
    baseline, feature_head = modules(exp)
    n = len(y_correction); heads = {}
    result = {"sp200_56k": np.r_[spline_cor, spline200_query]}
    for name, values in raw.items():
        model_frame = (frame_overrides or {}).get(name, frame)
        training = model_frame.iloc[:n].copy(); training["target"] = y_correction
        for kind in ["raw", "h29", "mat18", "per29"]:
            key = f"{kind}_{name}"
            if key not in REQUIRED: continue
            if kind == "raw": result[key] = values; continue
            columns = head18 if kind == "mat18" else head18 + SPECTRAL
            head_key = (name, len(columns))
            if head_key not in heads:
                fitted = feature_head.fit(training, values[:n], columns, 0.003, 0.5)
                corrected = baseline.corrected_probability(fitted, model_frame, values)[0]
                heads[head_key] = corrected
            corrected = heads[head_key]
            if kind == "h29": result[key] = corrected
            elif kind == "mat18": result[key] = .75 * corrected + .25 * np.r_[spline_cor, spline_mat]
            elif kind == "per29": result[key] = .75 * corrected + .25 * np.r_[spline_cor, spline_per]
    assert REQUIRED == set(result), sorted(REQUIRED - set(result))
    assert all(np.isfinite(v).all() and len(v) == len(frame) for v in result.values())
    return result


def route(cands, pops, y_correction, n_correction=6000, lam=.70):
    n_query = len(next(iter(cands.values()))) - n_correction
    raw = np.full(n_query, np.nan); shifted = np.full(n_query, np.nan); metadata = {}
    all_masks = np.column_stack([pops[name] for name in RECIPES])
    assert np.all(all_masks.sum(axis=1) == 1), "Population routing must partition every row"
    pi = float(np.mean(y_correction)); assert 0 < pi < 1
    for pop, names in RECIPES.items():
        mask_c = pops[pop][:n_correction]; mask_q = pops[pop][n_correction:]
        assert mask_c.any(), (pop, "No correction support")
        pc = np.mean([cands[name][:n_correction][mask_c] for name in names], axis=0)
        pq = np.mean([cands[name][n_correction:][mask_q] for name in names], axis=0)
        mean = float(np.mean(pc)); assert 0 < mean < 1
        displacement = float(lam * (logit(mean) - logit(pi)))
        raw[mask_q] = pq
        shifted[mask_q] = expit(logit(np.clip(pq, 1e-6, 1 - 1e-6)) - displacement)
        metadata[pop] = {"components": names, "correction_rows": int(mask_c.sum()), "query_rows": int(mask_q.sum()),
                         "mean_correction_prediction": mean, "global_correction_bad_rate": pi,
                         "logit_shift": displacement, "lambda": lam}
    assert np.isfinite(raw).all() and np.isfinite(shifted).all()
    return raw, shifted, metadata


def replay(home, output, folds):
    exp, frozen, joint = paths(home); baseline, feature_head = modules(exp)
    dependency_sources = [Path(__file__), Path(__file__).with_name("spline_models.py"),
                          exp / "transfer-20261025/source/baseline.py", exp / "transfer-20261025/source/feature_head.py"]
    plan = {"action": "fixed_recipe_correction_and_route_replay", "folds": folds,
            "source_sha256": {str(p): sha(p) for p in dependency_sources},
            "head_penalty": .003, "head_strength": .5, "head_fit_rows": 6000,
            "routing": RECIPES, "shift_lambda": .70,
            "reference_route_max_absolute_error_acceptance": 1e-10,
            "weights_saved": False, "scientific_selection_allowed": False}
    dump(output / "REPLAY_PLAN.json", plan)
    train_path = exp / "ranking-pipeline-20261020/input/data/train_features.parquet"
    train = pd.read_parquet(train_path, columns=["application_id", "target"])
    spectral_path = exp / "shopping-periodicity-fe-20261021/true_features.parquet"
    spectral = pd.read_parquet(spectral_path, columns=["application_id"] + SPECTRAL).set_index("application_id").loc[train.application_id]
    masks_path = joint / "raw_domain_features.npz"
    with np.load(masks_path, allow_pickle=True) as z:
        names = list(map(str, z["names"])); pop_global = {p: z["data"][:, names.index(p)] > .5 for p in RECIPES}
    head18 = json.loads((frozen / "frozen/fold0/MANIFEST.json").read_text())["head_columns"]
    old_route_path = joint / "routed_ALL66_peek.npy"
    old_route = np.load(old_route_path) if old_route_path.exists() else None
    info = []; pooled = np.full(70000, np.nan)
    for fold in folds:
        start = time.time(); root = output / f"replay/fold{fold}"; root.mkdir(parents=True, exist_ok=True)
        roles_path = frozen / f"frozen/fold{fold}/roles.npz"
        with np.load(roles_path, allow_pickle=False) as z: roles = {k: z[k] for k in z.files}
        want = np.r_[roles["correction"], roles["validation"]]
        frame_path = joint / f"fold{fold}/expanded.parquet"; frame = pd.read_parquet(frame_path)
        for column in SPECTRAL: frame[column] = spectral[column].to_numpy(dtype=np.float32)
        frame_q = frame.iloc[want].reset_index(drop=True)
        base_path = frozen / f"frozen/fold{fold}/baseline.npz"
        with np.load(base_path, allow_pickle=False) as z: raw_base = z["raw"]; sp_mat = z["spline"]
        # Exact historical reconstruction, including its head/baseline arithmetic.
        h29 = feature_head.fit(frame_q.iloc[:6000].assign(target=roles["y_correction"]), raw_base[:6000], head18 + SPECTRAL, .003, .5)
        corrected = baseline.corrected_probability(h29, frame_q.iloc[6000:], raw_base[6000:])[0]
        with np.load(exp / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=False) as z:
            old_periodic = z["periodicity__original"][roles["validation"]]
        with np.load(exp / f"transfer-20261025/spline_direct_ks/fold{fold}/validation.npz", allow_pickle=False) as z:
            sp_per = (old_periodic - .75 * corrected) / .25 + z["spline_delta"]
        with np.load(joint / "linear56k_runs.npz", allow_pickle=False) as z:
            sp_cor = z["oof_spline200"][fold][roles["correction"]]; sp200_q = z["val_spline200"][roles["validation"]]
        raws, source_files = raw_replay(exp, frozen, joint, fold, roles)
        cands = candidates(frame_q, raws, roles["y_correction"], head18, sp_cor, sp_mat, sp_per, sp200_q, exp)
        raw, shifted, routing = route(cands, {p: v[want] for p, v in pop_global.items()}, roles["y_correction"])
        error = float(np.max(np.abs(raw - old_route[roles["validation"]]))) if old_route is not None else None
        if error is not None: assert error < 1e-10, (fold, error)
        np.savez_compressed(root / "prediction.npz", rows=roles["validation"], raw=raw, prediction=shifted)
        record = {"fold": fold, "status": "passed", "seconds": time.time() - start, "raw_route_reference_max_absolute_error": error,
                  "raw_KS": ks(train.target.to_numpy()[roles["validation"]], raw),
                  "shifted_KS": ks(train.target.to_numpy()[roles["validation"]], shifted), "routing": routing,
                  "source_sha256": {str(p): sha(p) for p in [*dependency_sources, roles_path, frame_path, base_path, *source_files.values()]}}
        dump(root / "complete.json", record); info.append(record); pooled[roles["validation"]] = shifted
        print(json.dumps({k: record[k] for k in ["fold", "status", "seconds", "raw_route_reference_max_absolute_error", "raw_KS", "shifted_KS"]}), flush=True)
        del frame, frame_q, cands, raws; gc.collect()
    result = {"status": "passed", "folds": folds, "checks": info, "selection_status": "Reused development exploratory ALL66; not independent confirmation"}
    if np.isfinite(pooled).all():
        result["pooled_shifted_KS"] = ks(train.target.to_numpy(), pooled)
        np.savez_compressed(output / "replay/OOF.npz", prediction=pooled)
    dump(output / "replay/RESULT.json", result)


def spline200_columns(frozen, joint):
    import pyarrow.parquet as pq
    frozen_cols = pq.ParquetFile(frozen / "frozen/fold0/features.parquet").schema.names
    screen = pd.read_csv(joint / "inner_oof_residual_screen.csv")
    top = screen[~screen["in_champ_or_head"]].sort_values("max_abs_z", ascending=False)["name"].head(90).tolist()
    extras = ["v3_repay_180d_late_90_posterior_6", "v3_repay_180d_late_60_posterior_6",
              "v4_price_180d_between_category_std", "v4_price_all_between_category_std",
              "v4_price_90d_equal_category_mean", "v5_price_cancelled_all_equal_category_mean",
              "v5_ecom_180d_completion_posterior", "v5_price_income_residual",
              "v4_last12_late_90_posterior", "re_price_latent_multiplier",
              "re_income_log_measurement_sd", "re_income_log_sd",
              "re_late_1_posterior_sd", "re_late_30_posterior_sd", "joint_posterior_variance"]
    return list(dict.fromkeys(frozen_cols[:90] + frozen_cols[90:106] + frozen_cols[106:] + extras + top))


def fit_splines(home, output, fold, features):
    """Fixed production extension: fit spline branches on all70k, emit only test scores."""
    exp, frozen, joint = paths(home); baseline, _ = modules(exp)
    destination = output / f"splines/fold{fold}"; destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / "complete.json").exists(), "Preserve completed attempt; use a new output directory"
    frame = pd.read_parquet(features)
    train_path = exp / "ranking-pipeline-20261020/input/data/train_features.parquet"
    original = pd.read_parquet(train_path, columns=["application_id", "target"])
    apps_path = exp / "submission-full-periodicity-20261021/applications.parquet"
    apps = pd.read_parquet(apps_path, columns=["application_id"])
    assert len(frame) == 100000 and "target" not in frame and frame.application_id.tolist() == apps.application_id.tolist()
    assert frame.application_id.iloc[:70000].tolist() == original.application_id.tolist()
    # Historical fitted bundles were intentionally deleted to save storage.
    # Retained parameter metadata and the original source define the same columns.
    metadata_path = exp / f"ranking-pipeline-20261020/input/artifacts/fold{fold}/spline_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    mat_columns = metadata["recipe"]["spec"]["columns"]; per_columns = mat_columns + SPECTRAL
    assert len(mat_columns) == 77 and len(per_columns) == 88
    for col in frame[mat_columns+SPECTRAL].select_dtypes(include=["object","string","category"]).columns: frame[col] = frame[col].astype(object)
    full_columns = spline200_columns(frozen, joint)
    assert not set(mat_columns + per_columns + full_columns).difference(frame.columns), "Missing required fixed spline inputs"
    dependencies = [features, train_path, apps_path, metadata_path, Path(__file__), Path(__file__).with_name("spline_models.py")]
    plan = {"action": "fixed_70k_spline_production_refit", "fold": fold, "fit_rows": 70000, "test_rows": 30000,
            "source_sha256": {str(p): sha(p) for p in dependencies}, "mat77_columns": mat_columns,
            "per88_columns": per_columns, "sp200_columns": full_columns,
            "sp200_parameters": {"C": .0025, "n_iter": 4, "n_knots": 6, "degree": 2, "top_inter_n": 35},
            "fixed_spline_parameters": {"C": .05, "solver": "newton-cholesky", "max_iter": 80, "tol": 1e-5},
            "H016_parameters": {"strength": .5, "temperature": .5, "trust": 1., "ridge": .003,
                                 "maxiter": 300, "ftol": 1e-11, "gtol": 1e-7, "threshold": "logit(training_target_mean)"},
            "correction_shift_spline_policy": "Retain cached original56k innerOOF sp200 correction scores; use refitted70k branches for test.",
            "weights_saved": False, "independent_confirmation": False}
    dump(destination / "PLAN.json", plan)
    training = frame.iloc[:70000].copy(); training["target"] = original.target.to_numpy(int)
    query = frame.iloc[70000:].copy(); start = time.time()
    mat = fixed_spline_fit(training, mat_columns, baseline.prepare)
    mat_test = baseline.predict_spline(mat, query)
    del mat; gc.collect()
    per = fixed_spline_fit(training, per_columns, baseline.prepare)
    per_plain = baseline.predict_spline(per, query)
    per_test, h016 = direct_spline_update(per, training, query, baseline.prepare)
    del per; gc.collect()
    with threadpool_limits(limits=2):
        X = make_scaled_spline_matrix(frame, np.arange(70000), full_columns, n_knots=6, degree=2, top_inter_n=35)
        b0, weights = fit_logistic_irls(X[:70000], original.target.to_numpy(int), C=.0025)
        sp200_test, _ = predict_irls(X[70000:], b0, weights)
    for values in [mat_test, per_test, sp200_test]: assert len(values) == 30000 and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    prediction_path = destination / "prediction.npz"
    np.savez_compressed(prediction_path, rows=np.arange(70000,100000), application_id=query.application_id.to_numpy(str),
                        mat77=mat_test, per88_direct=per_test, per88_plain=per_plain, sp200=sp200_test)
    dump(destination / "complete.json", {"status": "passed", "fold": fold, "seconds": time.time()-start,
         "PLAN_sha256": sha(destination / "PLAN.json"), "predictions_sha256": sha(prediction_path),
         "H016_optimizer": h016, "weights_saved": False, "means": {"mat77": float(mat_test.mean()), "per88_direct": float(per_test.mean()), "sp200": float(sp200_test.mean())}})


def prediction_file(root, name, fold):
    aliases = {"champ_s7731": ["champ_s7731", "champion90_s7731", "kumo_large_champion90_s7731"],
               "shadow91": ["shadow91", "kumo_large_shadow91"], "pop_spec": ["pop_spec", "pop_specialist", "kumo_pop_specialist"]}
    candidates = [root / alias / f"fold{fold}" / filename for alias in aliases.get(name, [name]) for filename in ["prediction.npz", "pred.npz"]]
    return next((p for p in candidates if p.exists()), None)


def production_raw(root, fold, want, exp, frozen, joint, roles):
    result = {}; files = {}
    for name in BACKBONES:
        path = prediction_file(root, name, fold)
        if name == "kumo56":
            assert path is not None, "Missing all70k Kumo56 test predictions"
            test = align_prediction(path, want[6000:])
            old, old_files = raw_replay(exp, frozen, joint, fold, roles)
            result[name] = np.r_[old["kumo56"][:6000], test]
            files.update({str(p): sha(p) for k,p in old_files.items() if k.startswith("kumo56")})
        elif name == "pop_spec" and path is None:
            values = np.full(100000, np.nan)
            for sub in ["pop_spec_A", "pop_spec_BC1", "pop_spec_C2"]:
                part = prediction_file(root, sub, fold); assert part is not None, (sub, fold)
                with np.load(part, allow_pickle=False) as z: values[z["rows"]] = z["prediction"]
                files[str(part)] = sha(part)
            result[name] = values[want]
        else:
            assert path is not None, (name, fold, "Missing required backbone predictions")
            result[name] = align_prediction(path, want)
        if path is not None: files[str(path)] = sha(path)
    assert all(len(v) == 36000 and np.isfinite(v).all() for v in result.values())
    return result, files


def assemble_fold(home, output, fold, features, features70, prediction_root):
    exp, frozen, joint = paths(home)
    destination = output / f"production/fold{fold}"; destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / "complete.json").exists(), "Completed assembly is immutable; choose a new output directory"
    roles_path = frozen / f"frozen/fold{fold}/roles.npz"
    with np.load(roles_path, allow_pickle=False) as z: roles = {key:z[key] for key in z.files}
    want = np.r_[roles["correction"],np.arange(70000,100000)]
    frame = pd.read_parquet(features); frame70 = pd.read_parquet(features70)
    assert len(frame) == len(frame70) == 100000 and frame.application_id.tolist() == frame70.application_id.tolist()
    assert "target" not in frame and "target" not in frame70
    frame_q = frame.iloc[want].reset_index(drop=True)
    # Kumo56 historical correction raw scores came from nested OOF; pair them
    # with the original cached correction head feature values. Its test scores
    # use a new all70k context and corresponding all70k-prior test features.
    old_frame_path = joint / f"fold{fold}/expanded.parquet"
    old_frame = pd.read_parquet(old_frame_path)
    head18 = json.loads((frozen / "frozen/fold0/MANIFEST.json").read_text())["head_columns"]
    kumo56_head_frame = pd.concat([old_frame.iloc[roles["correction"]][head18],frame70.iloc[70000:][head18]],ignore_index=True)
    with np.load(joint / "linear56k_runs.npz", allow_pickle=False) as z: sp_cor = z["oof_spline200"][fold][roles["correction"]]
    spline_path = output / f"splines/fold{fold}/prediction.npz"
    spline_complete = output / f"splines/fold{fold}/complete.json"
    assert json.loads(spline_complete.read_text())["status"] == "passed"
    with np.load(spline_path, allow_pickle=False) as z:
        assert np.array_equal(z["rows"],want[6000:]); sp_mat=z["mat77"];sp_per=z["per88_direct"];sp200=z["sp200"]
    raw, raw_files = production_raw(prediction_root, fold, want, exp, frozen, joint, roles)
    plan = {"action": "fixed_ALL66_test_assembly", "fold": fold, "source_sha256": {str(p):sha(p) for p in [Path(__file__),features,features70,old_frame_path,roles_path,spline_path,spline_complete]},
            "raw_prediction_sha256": raw_files, "routing": RECIPES,"lambda":.70,"head_penalty":.003,"head_strength":.5,
            "correction_spline_policy":"Original56k innerOOF sp200; new70k test branches.",
            "Kumo56_policy":"Original nestedOOF correctionraw paired with original correctionheadfeatures; all70k test raw/headfeatures.",
            "scientific_selection_allowed":False,"weights_saved":False}
    dump(destination / "PLAN.json",plan)
    cands=candidates(frame_q,raw,roles["y_correction"],head18,sp_cor,sp_mat,sp_per,sp200,exp,frame_overrides={"kumo56":kumo56_head_frame})
    routed,shifted,routing=route(cands,{p:frame_q[p].to_numpy()>.5 for p in RECIPES},roles["y_correction"])
    output_path=destination/"prediction.npz"
    np.savez_compressed(output_path,rows=want[6000:],application_id=frame_q.application_id.iloc[6000:].to_numpy(str),raw=routed,prediction=shifted)
    dump(destination/"complete.json",{"status":"passed","fold":fold,"PLAN_sha256":sha(destination/"PLAN.json"),"prediction_sha256":sha(output_path),"routing":routing,"independent_confirmation":False})


def finish_csv(home,output):
    exp,_,_=paths(home)
    sample_path=exp/"submission-full-periodicity-20261021/sample_submission.csv";sample=pd.read_csv(sample_path)
    assert len(sample)==30000 and sample.application_id.is_unique
    values=[];source={}
    for fold in range(5):
        path=output/f"production/fold{fold}/prediction.npz";complete=path.with_name("complete.json")
        record=json.loads(complete.read_text());assert record["status"]=="passed"and record["prediction_sha256"]==sha(path)
        with np.load(path,allow_pickle=False)as z:
            ids=z["application_id"].astype(str);pred=z["prediction"]
            assert len(np.unique(ids))==30000 and set(ids)==set(sample.application_id.astype(str))
            values.append(pd.Series(pred,index=ids).reindex(sample.application_id.astype(str)).to_numpy())
        source[str(path)]=sha(path)
    final=np.mean(values,axis=0);assert np.isfinite(final).all()and((final>=0)&(final<=1)).all()
    result=pd.DataFrame({"application_id":sample.application_id,"default_probability":final})
    destination=output/"submission_ALL66_exploratory_shift070_full_refit.csv";result.to_csv(destination,index=False)
    dump(output/"CSV_VERIFICATION.json",{"status":"passed","rows":len(result),"unique_ids":30000,"sample_order_exact":True,"finite_bounded_predictions":True,"folds":5,"file":str(destination),"sha256":sha(destination),"prediction_sources":source,"sample_sha256":sha(sample_path),"selection_caveat":"ALL66 routing andlambda0.70 selected on reused OOF; no independently supported0.54 or public gain claim."})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["replay", "splines", "assemble", "finish"])
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_HOME / "logika-antigravity-refit-20261028/assembly")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--features70", type=Path)
    parser.add_argument("--prediction-root", type=Path)
    args = parser.parse_args()
    if args.action=="replay":replay(args.home, args.output, args.folds)
    elif args.action=="splines":
        assert args.features is not None;fit_splines(args.home,args.output,args.fold,args.features)
    elif args.action=="assemble":
        assert args.features is not None and args.features70 is not None and args.prediction_root is not None
        assemble_fold(args.home,args.output,args.fold,args.features,args.features70,args.prediction_root)
    elif args.action=="finish":finish_csv(args.home,args.output)


if __name__ == "__main__": main()
