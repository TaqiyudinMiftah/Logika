"""Matched feature, coverage and regularization contrasts for V07 and V101."""
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
import baseline as b
import feature_head
from study import fold_inputs, scores, emit


def read(path):
    return json.loads(Path(path).read_text())


def load(root):
    plan = read(root / "plan.json")
    inp = Path(plan["input"]); evidence = Path(plan["feature_evidence"])
    data = pd.read_parquet(inp / "data/train_features.parquet")
    base_columns = read(inp / "BASELINE_LOCK.json")["correction"]["features"]
    refs = {v: read(evidence / f"runs/{v}/RESULT.json") for v in ["V07", "V101"]}
    specs = {"base_original": dict(columns=base_columns, penalty=.003, strength=.5, view="base")}
    for v, reference in refs.items():
        for setting, penalty, strength in [
            ("original", .003, .5),
            ("published", reference["spec"]["penalty"], reference["spec"]["strength"]),
        ]:
            for treatment, columns in [
                ("base", base_columns),
                ("coverage", base_columns + reference["support_features"]),
                ("features", reference["columns"]),
            ]:
                if treatment == "base" and setting == "original":
                    continue
                name = f"{v}_{treatment}_{setting}"
                specs[name] = dict(columns=columns, penalty=penalty, strength=strength,
                                   view=v, treatment=treatment, setting=setting)
    assert len(specs) == 11
    return plan, inp, evidence, data, refs, specs


def frame_with_features(data, indices, extra):
    assert extra.application_id.tolist() == data.application_id.iloc[indices].tolist()
    new = [c for c in extra if c not in data]
    return pd.concat([data.iloc[indices].reset_index(drop=True),
                      extra[new].reset_index(drop=True)], axis=1)


def inputs(inp, evidence, data, fold):
    ix, cached = fold_inputs(inp, data, fold)
    frames = {"base": (data.iloc[ix["correction"]].copy(),
                       data.iloc[ix["validation"]].drop(columns="target").copy())}
    for v, prefix in [("V07", "data"), ("V101", "lifecycle/data")]:
        directory = evidence / f"{prefix}/fold{fold}"
        audit = read(directory / "AUDIT.json")
        assert audit["status"] == "PASS"
        assert audit["correction_ids_sha256"] == b.ids_hash(data, ix["correction"])
        assert audit["validation_ids_sha256"] == b.ids_hash(data, ix["validation"])
        train = frame_with_features(data, ix["correction"], pd.read_parquet(directory / "correction.parquet"))
        valid = frame_with_features(data, ix["validation"], pd.read_parquet(directory / "validation.parquet"))
        assert not set(train.columns) & {"prediction", "baseline_prediction"}
        frames[v] = train, valid.drop(columns="target")
    return ix, cached, frames


def check_deadline(root):
    state = read(root / "state.json")
    assert time.time() < state["deadline"] and not (root / "STOP").exists()


def preflight(root):
    plan, inp, evidence, data, refs, specs = load(root)
    imported = read(evidence / "import_manifest.json")
    for name, digest in imported["files"].items():
        assert b.sha(evidence / name) == digest, name
    seen = np.zeros(len(data), int)
    replayed = {v: np.full(len(data), np.nan) for v in refs}
    baseline = np.full(len(data), np.nan)
    checks = []
    for fold in range(5):
        ix, p, frames = inputs(inp, evidence, data, fold)
        baseline[ix["validation"]] = p["final_oof"]
        seen[ix["validation"]] += 1
        for v in refs:
            model = joblib.load(evidence / f"runs/{v}/fold{fold}_correction.joblib")
            assert model["columns"] == refs[v]["columns"]
            assert not set(model["columns"]) & {"target", "customer_id", "application_id", "split"}
            train, valid = frames[v]
            actual = .75 * b.corrected_probability(model, valid, p["kumo_oof"])[0] + .25 * p["spline_oof"]
            saved = pd.read_parquet(evidence / f"runs/{v}/fold{fold}_validation.parquet")
            assert saved.application_id.tolist() == valid.application_id.tolist()
            error = float(np.max(abs(actual - saved.prediction)))
            assert error < 1e-12, (v, fold, error)
            poisoned = valid.iloc[:20].copy()
            poisoned["target"] = 999
            poisoned["customer_id"] = "POISONED"
            repeated = .75 * b.corrected_probability(model, poisoned, p["kumo_oof"][:20])[0] + .25 * p["spline_oof"][:20]
            assert np.array_equal(actual[:20], repeated)
            replayed[v][ix["validation"]] = actual
            checks.append(dict(fold=fold, recipe=v, replay_max_error=error))
    assert np.all(seen == 1)
    y = data.target.to_numpy(int)
    metrics = {v: scores(y, pred, independent=True) for v, pred in replayed.items()}
    metrics["incumbent"] = scores(y, baseline, independent=True)
    for v in refs:
        for metric in ["KS", "AUC", "logloss"]:
            assert abs(metrics[v][metric] - refs[v]["OOF"][metric]) < 1e-12
    assert abs(metrics["incumbent"]["KS"] - .5292994659005608) < 1e-12
    # Test the actual newly extracted estimator against the known control on real rows.
    ix, p, frames = inputs(inp, evidence, data, 0)
    train, valid = frames["base"]
    model = feature_head.fit(train, p["kumo_holdout"], specs["base_original"]["columns"], .003, .5)
    prediction = .75 * b.corrected_probability(model, valid, p["kumo_oof"])[0] + .25 * p["spline_oof"]
    assert np.max(abs(prediction - p["final_oof"])) < 2e-6
    root.joinpath("checks").mkdir(exist_ok=True)
    joblib.dump(model, root / "checks/control_refit.joblib", compress=3)
    result = dict(status="passed", replayed=checks, metrics=metrics,
                  feature_files_verified=len(imported["files"]),
                  preflight_model_fits=1, planned_main_fits=55,
                  control_refit_max_error=float(np.max(abs(prediction - p["final_oof"]))),
                  feature_sources_independently_rebuilt=False,
                  outer_label_poisoning="passed", historical_labels_reused=True)
    b.dump(result, root / "preflight.json")
    return result


def fit_fold(root, fold):
    check_deadline(root)
    plan, inp, evidence, data, refs, specs = load(root)
    ix, p, frames = inputs(inp, evidence, data, fold)
    dest = root / f"folds/fold{fold}"; dest.mkdir(parents=True, exist_ok=False)
    predictions, diagnostics = {}, {}
    started = time.time()
    for name, spec in specs.items():
        check_deadline(root)
        train, valid = frames[spec["view"]]
        model = feature_head.fit(train, p["kumo_holdout"], spec["columns"],
                                 spec["penalty"], spec["strength"])
        joblib.dump(model, dest / (name + ".joblib"), compress=3)
        head, delta = b.corrected_probability(model, valid, p["kumo_oof"])
        pred = .75 * head + .25 * p["spline_oof"]
        assert np.array_equal(head, b.corrected_probability(
            joblib.load(dest / (name + ".joblib")), valid, p["kumo_oof"])[0])
        if name == "base_original":
            assert np.max(abs(pred - p["final_oof"])) < 2e-6
        for v in refs:
            if name == f"{v}_features_published":
                original = pd.read_parquet(evidence / f"runs/{v}/fold{fold}_validation.parquet")
                assert np.max(abs(pred - original.prediction)) < 2e-6, (name, fold)
        predictions[name] = pred
        fitted, _ = b.corrected_probability(model, train, p["kumo_holdout"])
        # Training-only diagnostics now; outer metrics are calculated after all fits.
        diagnostics[name] = dict(
            parameters=len(model["coefficients"]), columns=spec["columns"],
            penalty=spec["penalty"], strength=spec["strength"],
            training_corrected_Kumo=scores(train.target.to_numpy(int), fitted),
            mean_absolute_applied_logit_correction=float(np.mean(abs(delta * spec["strength"]))),
            p99_absolute_applied_logit_correction=float(np.quantile(abs(delta * spec["strength"]), .99)),
            gradient_max=model["gradient_max_abs"],
        )
        emit("recipe_complete", fold=fold, recipe=name)
    predictions["incumbent"] = p["final_oof"]
    np.savez_compressed(dest / "predictions.npz", rows=ix["validation"], **predictions)
    b.dump(diagnostics, dest / "diagnostics.json")
    b.dump(dict(
        fits=len(specs), seconds=time.time() - started,
        context_ids_sha=b.ids_hash(data, ix["context"]),
        correction_ids_sha=b.ids_hash(data, ix["correction"]),
        validation_ids_sha=b.ids_hash(data, ix["validation"]),
        files={path.name: b.sha(path) for path in dest.iterdir() if path.is_file()},
    ), dest / "complete.json")
    emit("outer_complete", fold=fold, fits=len(specs))


def finish(root):
    check_deadline(root)
    plan, inp, evidence, data, refs, specs = load(root)
    y = data.target.to_numpy(int)
    predictions = {name: np.full(len(y), np.nan) for name in [*specs, "incumbent"]}
    folds = np.full(len(y), -1); seen = np.zeros(len(y), int); verified = 0
    for fold in range(5):
        dest = root / f"folds/fold{fold}"
        for name, digest in read(dest / "complete.json")["files"].items():
            assert b.sha(dest / name) == digest
            verified += 1
        z = np.load(dest / "predictions.npz"); rows = z["rows"]
        ix, p, frames = inputs(inp, evidence, data, fold)
        assert np.array_equal(rows, ix["validation"])
        for name, spec in specs.items():
            model = joblib.load(dest / (name + ".joblib"))
            replay = .75 * b.corrected_probability(model, frames[spec["view"]][1],
                                                   p["kumo_oof"])[0] + .25 * p["spline_oof"]
            assert np.array_equal(replay, z[name])
        folds[rows] = fold; seen[rows] += 1
        for name in predictions:
            predictions[name][rows] = z[name]
    assert np.all(seen == 1)
    metrics = {name: scores(y, p, independent=True) for name, p in predictions.items()}
    fold_metrics = {name: [scores(y[folds == f], p[folds == f], independent=True)
                           for f in range(5)] for name, p in predictions.items()}
    contrasts = {}
    for v in refs:
        definitions = {
            "regularization": (f"{v}_base_published", "base_original"),
            "coverage_published": (f"{v}_coverage_published", f"{v}_base_published"),
            "features_published": (f"{v}_features_published", f"{v}_coverage_published"),
            "coverage_original": (f"{v}_coverage_original", "base_original"),
            "features_original": (f"{v}_features_original", f"{v}_coverage_original"),
            "total_published": (f"{v}_features_published", "incumbent"),
        }
        for label, (candidate, control) in definitions.items():
            difference = {key: metrics[candidate][key] - metrics[control][key]
                          for key in ["KS", "AUC", "logloss", "average_precision"]}
            fd = [{key: fold_metrics[candidate][f][key] - fold_metrics[control][f][key]
                   for key in ["KS", "AUC", "logloss"]} for f in range(5)]
            contrasts[v + "_" + label] = dict(
                candidate=candidate, control=control, delta=difference, folds=fd,
                mean_fold_KS=float(np.mean([m["KS"] for m in fd])),
                KS_fold_wins=sum(m["KS"] > 0 for m in fd),
                AUC_fold_wins=sum(m["AUC"] > 0 for m in fd),
            )
    # Weighted bootstrap: one fixed ordering per prediction vector, exact tied-score CDFs.
    structures = {}
    for name, p in predictions.items():
        order = np.argsort(p, kind="stable")
        end = np.r_[p[order][1:] != p[order][:-1], True]
        structures[name] = order, end
    groups = [np.flatnonzero((folds == f) & (y == c)) for f in range(5) for c in [0, 1]]
    rng = np.random.default_rng(20261023)
    draws = {name: [] for name in contrasts}
    for rep in range(plan["bootstrap_repetitions"]):
        if rep % 100 == 0:
            check_deadline(root); emit("bootstrap", done=rep, total=plan["bootstrap_repetitions"])
        weight = np.zeros(len(y), int)
        for g in groups:
            weight[g] = rng.multinomial(len(g), np.full(len(g), 1 / len(g)))
        n1, n0 = weight[y == 1].sum(), weight[y == 0].sum()
        sampled = {}
        for name, (order, end) in structures.items():
            cp = np.cumsum(weight[order] * (y[order] == 1))[end]
            cn = np.cumsum(weight[order] * (y[order] == 0))[end]
            dp = np.diff(np.r_[0, cp]); dn = np.diff(np.r_[0, cn])
            sampled[name] = np.array([np.max(abs(cp / n1 - cn / n0)),
                                     np.sum(dp * (cn - .5 * dn)) / (n1 * n0)])
        for name, contrast in contrasts.items():
            draws[name].append(sampled[contrast["candidate"]] - sampled[contrast["control"]])
    for name, contrast in contrasts.items():
        bounds = np.quantile(draws[name], [.025, .975], axis=0)
        contrast["conditional_95_interval"] = dict(KS=bounds[:, 0].tolist(), AUC=bounds[:, 1].tolist())
        d = contrast["delta"]
        contrast["point_direction"] = ("both_positive" if d["KS"] > 0 and d["AUC"] > 0
                                      else "mixed_or_nonpositive")
    np.savez_compressed(root / "oof.npz", target=y, fold=folds, **predictions)
    np.savez_compressed(root / "bootstrap.npz", **{k: np.asarray(v) for k, v in draws.items()})
    result = dict(
        status="complete", recipes=len(specs), main_correction_fits=5 * len(specs),
        verified_hashes=verified, metrics=metrics, fold_metrics=fold_metrics, contrasts=contrasts,
        selected_winner=None, submission_generated=False, historical_labels_reused=True,
        point_in_time_certified=False,
        uncertainty="Post-selection descriptive paired fold/class bootstrap; unadjusted marginal "
        "intervals, conditional on fitted predictions; not confirmation or a private leaderboard.",
    )
    b.dump(result, root / "result.json")
    pd.DataFrame([dict(recipe=name, **m) for name, m in metrics.items()]).to_csv(root / "scores.csv", index=False)
    pd.DataFrame([dict(contrast=name, candidate=c["candidate"], control=c["control"],
                       **c["delta"], KS_fold_wins=c["KS_fold_wins"], mean_fold_KS=c["mean_fold_KS"])
                  for name, c in contrasts.items()]).to_csv(root / "feature_effects.csv", index=False)
    lines = ["# Feature versus regularization attribution", "",
             "The 11 fixed recipes completed five outer folds. No post-hoc winner was selected.", "",
             "| Contrast | ΔKS | ΔAUC | Δlog loss | KS fold wins |",
             "|---|---:|---:|---:|---:|"]
    for name, contrast in contrasts.items():
        d = contrast["delta"]
        lines.append(f"| {name} | {d['KS']:+.8f} | {d['AUC']:+.8f} | {d['logloss']:+.8f} | {contrast['KS_fold_wins']}/5 |")
    lines += ["", "Read `features_published` to assess the feature block beyond matched regularization and coverage.",
              "Read `regularization` and `coverage_published` to explain the other parts of its total change.",
              "The original-setting feature contrast tests whether this attribution depends on regularization.",
              "", "All 55 new heads are retained and replayed against the saved Kumo scores.",
              "Published V07/V101 predictions were reconstructed before the experiment.",
              "These families were chosen using historical results; all intervals are descriptive.",
              "No independent fresh data, public-score guarantee, or new competition submission is claimed."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    emit("DONE", status="complete", report=str(root / "REPORT.md"))


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
