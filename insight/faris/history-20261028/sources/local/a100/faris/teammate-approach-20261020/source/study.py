"""Train and assess a small conditional correction study, without submissions."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import log_loss, average_precision_score
from sklearn.model_selection import StratifiedKFold
import baseline as b
import joint_model as jm


def read(path):
    return json.loads(Path(path).read_text())


def emit(stage, **values):
    print(json.dumps(dict(time=time.time(), stage=stage, **values)), flush=True)


def check_deadline(root):
    state = read(root / "state.json")
    assert time.time() < state["deadline"] and not (root / "STOP").exists()


def load(root):
    plan = read(root / "plan.json")
    source = Path(plan["input"])
    data = pd.read_parquet(source / "data/train_features.parquet")
    spec = read(source / "provenance/definition.json")
    assert len(data) == 70000 and data.application_id.is_unique
    return plan, source, data, spec


def scores(y, p, independent=False):
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    result = b.scores(y, p)
    result.update(logloss=float(log_loss(y, p)),
                  average_precision=float(average_precision_score(y, p)))
    if independent:
        assert abs(result["KS"] - ks_2samp(p[y == 1], p[y == 0]).statistic) < 1e-12
        n1 = y.sum()
        independent_auc = (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
        assert abs(result["AUC"] - independent_auc) < 1e-12
    return result


def fold_inputs(source, data, fold):
    directory = source / f"artifacts/fold{fold}"
    indices = dict(np.load(directory / "indices.npz"))
    meta = read(directory / "metadata.json")
    for name, label in [("train", "training"), ("validation", "validation"),
                        ("context", "context"), ("correction", "correction_training")]:
        assert b.ids_hash(data, indices[name]) == meta[label + "_ids_sha256"]
    assert len(indices["context"]) == 50000 and len(indices["correction"]) == 6000
    assert not np.intersect1d(indices["context"], indices["correction"]).size
    assert not np.intersect1d(indices["train"], indices["validation"]).size
    assert np.array_equal(np.sort(np.r_[indices["context"], indices["correction"]]), indices["train"])
    predictions = {}
    for filename, group in [("kumo_holdout", "correction"), ("kumo_oof", "validation"),
                            ("spline_oof", "validation"), ("final_oof", "validation")]:
        frame = pd.read_parquet(directory / (filename + ".parquet"))
        assert frame.application_id.tolist() == data.application_id.iloc[indices[group]].tolist()
        assert np.array_equal(frame.target.to_numpy(), data.target.iloc[indices[group]].to_numpy())
        predictions[filename] = frame.prediction.to_numpy(float)
    return indices, predictions


def preflight(root):
    plan, source, data, spec = load(root)
    seen = np.zeros(len(data), int)
    prediction = np.full(len(data), np.nan)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=31027).split(data, data.target))
    for f in range(5):
        ix, p = fold_inputs(source, data, f)
        assert np.array_equal(ix["train"], folds[f][0])
        assert np.array_equal(ix["validation"], folds[f][1])
        seen[ix["validation"]] += 1
        original = joblib.load(source / f"artifacts/fold{f}/correction.joblib")
        validation = data.iloc[ix["validation"]].drop(columns="target")
        replay = .75 * b.corrected_probability(original, validation, p["kumo_oof"])[0] + .25 * p["spline_oof"]
        assert np.max(abs(replay - p["final_oof"])) < 1e-12
        prediction[ix["validation"]] = replay
    assert np.all(seen == 1)
    actual = scores(data.target.to_numpy(), prediction, independent=True)
    assert abs(actual["KS"] - .5292994659005608) < 1e-12
    assert abs(actual["AUC"] - .8472026883917805) < 1e-12
    # Analytic objective gradient and tied-probability metric checks.
    rng = np.random.default_rng(713)
    x = rng.normal(size=(30, 8)); y = rng.integers(0, 2, 30)
    off = rng.normal(size=30); penalties = np.r_[0., np.full(7, .03)]
    beta = rng.normal(size=8) / 10
    error = check_grad(lambda z: jm.objective(z, x, off, y, penalties)[0],
                       lambda z: jm.objective(z, x, off, y, penalties)[1], beta)
    assert error < 1e-6
    scores(np.array([0, 1, 0, 1]), np.array([.2, .2, .8, .8]), independent=True)
    # Real-data fitted-pipeline probe, with evaluation-label poisoning.
    ix, p = fold_inputs(source, data, 0)
    train = data.iloc[ix["correction"][:1200]]
    raw = p["kumo_holdout"][:1200]
    additive = b.refit_correction(spec, train, raw)
    fitted = jm.fit(additive, train, raw, "measurement_reliability")
    evaluation = data.iloc[ix["correction"][1200:1220]].copy()
    pred = jm.predict(fitted, evaluation, p["kumo_holdout"][1200:1220])
    evaluation["target"] = 999
    evaluation["application_id"] = "POISONED_ID"
    assert np.array_equal(pred, jm.predict(fitted, evaluation, p["kumo_holdout"][1200:1220]))
    root.joinpath("checks").mkdir(exist_ok=True)
    joblib.dump(fitted, root / "checks/probe.joblib", compress=3)
    assert np.array_equal(pred, jm.predict(joblib.load(root / "checks/probe.joblib"),
                                         evaluation, p["kumo_holdout"][1200:1220]))
    result = dict(status="passed", baseline=actual, gradient_error=error,
                  eval_label_and_id_poisoning="passed", fitted_replay="passed",
                  strict_new_label_boundary=True, historical_label_reuse=True,
                  inherited_point_in_time_certification=False)
    b.dump(result, root / "preflight.json")
    return result


def save_model(model, path):
    joblib.dump(model, path, compress=3)
    return b.sha(path)


def residual_diagnostics(frame, control_oof):
    """Descriptive evidence on correction rows, never a new routing rule."""
    result = {}
    for name, pairs in jm.PAIRS.items():
        for left, right in pairs:
            a, c = frame[left].to_numpy(float), frame[right].to_numpy(float)
            qa = np.nanquantile(a, [1 / 3, 2 / 3])
            qc = np.nanquantile(c, [1 / 3, 2 / 3])
            ia, ic = np.digitize(a, qa), np.digitize(c, qc)
            valid = np.isfinite(a) & np.isfinite(c)
            cells = []
            for i in range(3):
                for j in range(3):
                    take = valid & (ia == i) & (ic == j)
                    if not take.any():
                        continue
                    residual = frame.target.to_numpy()[take] - control_oof[take]
                    cells.append(dict(
                        left_bin=i, right_bin=j, rows=int(take.sum()),
                        mean_residual=float(residual.mean()),
                        residual_standard_error=float(residual.std() / np.sqrt(len(residual))),
                    ))
            result[f"{name}:{left}:{right}"] = dict(
                left_cuts=qa.tolist(), right_cuts=qc.tolist(), cells=cells,
                missing_rows=int((~valid).sum()), used_for_selection=False,
                interval_caveat="Descriptive training residuals; no multiplicity-adjusted inference.",
            )
    return result


def fit_fold(root, fold):
    check_deadline(root)
    plan, source, data, spec = load(root)
    ix, p = fold_inputs(source, data, fold)
    out = root / f"folds/fold{fold}"
    out.mkdir(parents=True, exist_ok=False)
    ci, vi = ix["correction"], ix["validation"]
    train = data.iloc[ci].copy()
    # No outer labels are passed to any feature builder, head or selector.
    validation = data.iloc[vi].drop(columns="target")
    raw = p["kumo_holdout"]
    names = list(jm.PAIRS)
    inner_predictions = {name: np.full(len(ci), np.nan) for name in names}
    seen = np.zeros(len(ci), int)
    boundaries = []
    inner = StratifiedKFold(3, shuffle=True, random_state=plan["inner_seed"] + fold)
    for j, (ti, ei) in enumerate(inner.split(train, train.target)):
        check_deadline(root)
        assert not np.intersect1d(ci[ti], ci[ei]).size
        assert not np.intersect1d(ci[ei], ix["context"]).size
        dest = out / f"inner{j}"; dest.mkdir()
        additive = b.refit_correction(spec, train.iloc[ti], raw[ti])
        for name in names:
            model = jm.fit(additive, train.iloc[ti], raw[ti], name)
            inner_predictions[name][ei] = jm.predict(
                model, train.iloc[ei].drop(columns="target"), raw[ei])
            save_model(model, dest / (name + ".joblib"))
        seen[ei] += 1
        np.savez_compressed(dest / "indices.npz", train=ci[ti], validation=ci[ei])
        boundaries.append(dict(inner=j, fit=len(ti), evaluation=len(ei),
                               fitting_ids_sha=b.ids_hash(data, ci[ti]),
                               evaluation_ids_sha=b.ids_hash(data, ci[ei])))
        emit("inner_complete", outer_fold=fold, inner_fold=j)
    assert np.all(seen == 1)
    inner_scores = {name: scores(train.target.to_numpy(), pred)
                    for name, pred in inner_predictions.items()}
    b.dump(residual_diagnostics(train, inner_predictions["control"]),
           out / "residual_diagnostics.json")
    eligible = [name for name in names
                if inner_scores[name]["AUC"] >= inner_scores["control"]["AUC"]]
    selected = min(eligible, key=lambda name: (inner_scores[name]["logloss"], names.index(name)))
    # Choice is written before final validation predictions or metrics.
    b.dump(dict(selected=selected, metrics=inner_scores, eligible=eligible,
                selection="minimum inner OOF logloss with nondecreasing AUC versus control",
                selected_using_outer_labels=False, boundaries=boundaries),
           out / "selection.json")
    np.savez_compressed(out / "inner_oof.npz", rows=ci, target=train.target.to_numpy(),
                        **inner_predictions)
    additive = b.refit_correction(spec, train, raw)
    predictions, replay_errors = {}, {}
    for name in names:
        check_deadline(root)
        model = jm.fit(additive, train, raw, name)
        filename = out / (name + ".joblib")
        save_model(model, filename)
        head = jm.predict(model, validation, p["kumo_oof"])
        repeated = jm.predict(joblib.load(filename), validation, p["kumo_oof"])
        replay_errors[name] = float(np.max(abs(head - repeated)))
        assert replay_errors[name] == 0.
        predictions[name] = .75 * head + .25 * p["spline_oof"]
    refit_error = float(np.max(abs(predictions["control"] - p["final_oof"])))
    assert refit_error < 2e-6, refit_error
    # Compare to exact saved incumbent; refitted control is a separate numeric check.
    predictions["incumbent"] = p["final_oof"]
    predictions["primary"] = predictions[selected]
    np.savez_compressed(out / "outer_predictions.npz", rows=vi, **predictions)
    b.dump(dict(selected=selected, refit_max_abs_error=refit_error,
                replay_max_errors=replay_errors, correction_fits=16,
                complete_oof_fold=True,
                files={str(path.relative_to(out)): b.sha(path)
                       for path in out.rglob("*") if path.is_file()}),
           out / "complete.json")
    emit("outer_complete", fold=fold, selected=selected, fitted_corrections=16)


def calibration(y, p):
    # Equal-count deciles for descriptive probability diagnostics, not threshold tuning.
    rows = []
    for j, ix in enumerate(np.array_split(np.argsort(p, kind="stable"), 10)):
        rows.append(dict(decile=j + 1, rows=len(ix), predicted=float(p[ix].mean()),
                         observed=float(y[ix].mean())))
    return rows


def finish(root):
    check_deadline(root)
    plan, source, data, spec = load(root)
    y = data.target.to_numpy(int)
    names = ["incumbent", "control", "primary"] + [n for n in jm.PAIRS if n != "control"]
    predictions = {name: np.full(len(y), np.nan) for name in names}
    seen = np.zeros(len(y), int); fold_ids = np.full(len(y), -1)
    choices, file_count = [], 0
    for f in range(5):
        dest = root / f"folds/fold{f}"
        meta = read(dest / "complete.json")
        for name, digest in meta["files"].items():
            assert b.sha(dest / name) == digest, name
            file_count += 1
        z = np.load(dest / "outer_predictions.npz")
        ix, raw = fold_inputs(source, data, f)
        rows = z["rows"]
        assert np.array_equal(rows, ix["validation"])
        seen[rows] += 1; fold_ids[rows] = f
        for name in names:
            predictions[name][rows] = z[name]
        choices.append(read(dest / "selection.json"))
        for name in jm.PAIRS:
            model = joblib.load(dest / (name + ".joblib"))
            replay = .75 * jm.predict(model, data.iloc[rows].drop(columns="target"),
                                       raw["kumo_oof"]) + .25 * raw["spline_oof"]
            assert np.array_equal(replay, z[name])
    assert np.all(seen == 1)
    metrics = {name: scores(y, p, independent=True) for name, p in predictions.items()}
    folded = {name: [scores(y[fold_ids == f], p[fold_ids == f], independent=True)
                     for f in range(5)] for name, p in predictions.items()}
    assert abs(metrics["incumbent"]["KS"] - .5292994659005608) < 1e-12
    comparisons = ["primary"] + [n for n in jm.PAIRS if n != "control"]
    rng = np.random.default_rng(plan["bootstrap_seed"])
    groups = [np.flatnonzero((fold_ids == f) & (y == c)) for f in range(5) for c in [0, 1]]
    draws = {name: [] for name in comparisons}
    for rep in range(plan["bootstrap_repetitions"]):
        if rep % 100 == 0:
            check_deadline(root); emit("uncertainty", completed=rep, total=plan["bootstrap_repetitions"])
        rows = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        base = b.scores(y[rows], predictions["incumbent"][rows])
        for name in comparisons:
            m = b.scores(y[rows], predictions[name][rows])
            draws[name].append([m["KS"] - base["KS"], m["AUC"] - base["AUC"]])
    intervals = {}
    for name, sample in draws.items():
        alpha = .05 if name == "primary" else .05 / 3
        intervals[name] = dict(
            delta_KS=np.quantile(np.asarray(sample)[:, 0], [alpha / 2, 1 - alpha / 2]).tolist(),
            delta_AUC=np.quantile(np.asarray(sample)[:, 1], [alpha / 2, 1 - alpha / 2]).tolist(),
            coverage_per_metric=1 - alpha,
        )
    np.savez_compressed(root / "oof.npz", rows=np.arange(len(y)), target=y, fold=fold_ids, **predictions)
    np.savez_compressed(root / "bootstrap.npz", **{k: np.asarray(v) for k, v in draws.items()})
    deltas = {name: {metric: metrics[name][metric] - metrics["incumbent"][metric]
                    for metric in ["KS", "AUC", "logloss", "average_precision"]}
              for name in comparisons}
    primary_delta = deltas["primary"]
    label = ("point_improvement_requires_further_assessment"
             if primary_delta["KS"] > 0 and primary_delta["AUC"] > 0
             else "no_joint_KS_AUC_improvement")
    result = dict(
        status="complete", conclusion=label, primary="inner_selected_procedure",
        metrics=metrics, deltas=deltas, fold_metrics=folded, intervals=intervals,
        inner_choices=[x["selected"] for x in choices], correction_fits=80,
        research_hypotheses=3, oof_rows=70000, inspected_artifact_hashes=file_count,
        historical_label_reuse=True, inherited_global_feature_references=True,
        point_in_time_certified=False, automatic_submission=False,
        uncertainty_scope="Paired fold/class bootstrap conditional on fitted predictions; "
        "primary marginal 95% intervals; secondary three-comparison Bonferroni per metric; "
        "historical adaptive selection and retraining uncertainty excluded.",
    )
    b.dump(result, root / "result.json")
    b.dump({name: calibration(y, pred) for name, pred in predictions.items()},
           root / "calibration.json")
    pd.DataFrame([dict(pipeline=name, **metrics[name]) for name in names]).to_csv(root / "scores.csv", index=False)
    lines = ["# Conditional correction research", "", f"Conclusion: {label}.", "",
             "The primary procedure chooses its correction using only inner training validation.",
             "The three fixed variants below are secondary diagnostics, not post-hoc winners.", "",
             "| Pipeline | KS | AUC | Log loss |", "|---|---:|---:|---:|"]
    for name in names:
        m = metrics[name]
        lines.append(f"| {name} | {m['KS']:.8f} | {m['AUC']:.8f} | {m['logloss']:.8f} |")
    lines += ["", "Primary choices by outer fold: " + ", ".join(result["inner_choices"]) + ".",
              "", f"Primary KS change: {primary_delta['KS']:+.8f}; AUC change: {primary_delta['AUC']:+.8f}.",
              f"KS interval: {intervals['primary']['delta_KS']}.",
              f"AUC interval: {intervals['primary']['delta_AUC']}.", "",
              "80 correction fits implement three hypotheses and one nested selector across five folds.",
              "All models, inner predictions and final OOF predictions are retained.",
              "No new Kumo inference or competition submission was needed for this matched correction comparison.",
              "All labels have historical exposure; these are not private leaderboard scores or untouched confirmation.",
              "Snapshot timing uncertainty and globally estimated label-free reference features are inherited.",
              "Confidence intervals omit historical selection and retraining uncertainty."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    b.dump(dict(
        source_input=str(source), source_lock_sha=b.sha(source / "BASELINE_LOCK.json"),
        formula=".75*corrected_Kumo+.25*original_spline; average five fold pipelines for test",
        folds=[dict(fold=f, selected=choices[f]["selected"],
                    path=f"folds/fold{f}/{choices[f]['selected']}.joblib",
                    sha=b.sha(root / f"folds/fold{f}/{choices[f]['selected']}.joblib"))
               for f in range(5)],
        backbone="unchanged original Kumo context; raw-score caches were verified",
        competition_submission_generated=False,
    ), root / "fitted_pipeline.json")
    emit("DONE", status="complete", conclusion=label, report=str(root / "REPORT.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "fold", "finish"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(args.root), indent=2))
    elif args.action == "fold":
        fit_fold(args.root, args.fold)
    else:
        finish(args.root)
