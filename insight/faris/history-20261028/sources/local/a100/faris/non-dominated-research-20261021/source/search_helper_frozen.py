#!/usr/bin/env python3
"""NSGA-II-inspired design calculator. UNTESTED: written without execution.

Standard-library only. Computes descriptors/ranks/admission proposals; it does
not train, evaluate a model, certify validation, or update input archives.
See ../references/calculator.md for the input contract and CLI.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys


VERSION = 1
COHORT_KEYS = (
    "protocol_sha256", "data_sha256", "split_sha256", "evaluator_sha256",
    "metric", "direction", "fidelity", "aggregation_id",
)
STATUSES = {"proposed", "pending", "running", "complete", "failed",
            "pruned", "interrupted"}
ORIGINS = {"ai_hypothesis", "mutation", "crossover", "baseline", "verification"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_value(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON number")
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)


def load(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    obj = json.loads(raw, parse_constant=reject_constant,
                     object_pairs_hook=unique_pairs)
    finite_tree(obj)
    return obj, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


class Schema:
    def __init__(self, raw, identity):
        require(isinstance(raw, dict) and raw.get("schema_version") == VERSION,
                "unsupported descriptor schema version")
        require(text_value(raw.get("id")), "descriptor schema needs an id")
        self.raw = raw
        self.identity = identity
        self.groups = raw.get("groups")
        require(isinstance(self.groups, list) and self.groups,
                "schema needs nonempty groups")
        self.fields = {}
        group_names = set()
        for group in self.groups:
            require(isinstance(group, dict) and text_value(group.get("name")),
                    "each group needs a name")
            require(group["name"] not in group_names, "duplicate group name")
            group_names.add(group["name"])
            weight = group.get("weight", 1.0)
            require(number(weight) and weight > 0, "group weight must be positive")
            fields = group.get("fields")
            require(isinstance(fields, list) and fields, "empty descriptor group")
            for field in fields:
                require(isinstance(field, dict) and text_value(field.get("name")),
                        "each field needs a name")
                name = field["name"]
                require(name not in self.fields, f"duplicate field: {name}")
                kind = field.get("type")
                require(kind in {"number", "categorical", "set"},
                        f"unsupported field type: {name}")
                when = field.get("when", {})
                require(isinstance(when, dict), f"invalid when: {name}")
                for parent, allowed in when.items():
                    require(parent in self.fields and
                            self.fields[parent]["type"] == "categorical",
                            f"{name}: conditions must reference earlier categorical fields")
                    require(isinstance(allowed, list) and allowed and
                            all(isinstance(x, str) for x in allowed),
                            f"{name}: condition values must be a nonempty string list")
                    parent_choices = self.fields[parent].get("choices")
                    if parent_choices is not None:
                        require(set(allowed) <= set(parent_choices),
                                f"{name}: unknown condition choice")
                if kind == "number":
                    low, high = field.get("min"), field.get("max")
                    require(number(low) and number(high) and low < high,
                            f"{name}: finite increasing bounds required")
                    require(float(low) < float(high),
                            f"{name}: bounds indistinguishable at floating precision")
                    scale = field.get("scale", "linear")
                    require(scale in {"linear", "log"}, f"{name}: unknown scale")
                    require(scale != "log" or low > 0,
                            f"{name}: log bounds must be positive")
                    if scale == "log":
                        require(math.log(low) < math.log(high),
                                f"{name}: bounds indistinguishable in log space")
                if "choices" in field:
                    choices = field["choices"]
                    require(kind in {"categorical", "set"} and
                            isinstance(choices, list) and choices and
                            all(isinstance(x, str) for x in choices) and
                            len(set(choices)) == len(choices),
                            f"{name}: invalid choices")
                self.fields[name] = field

    def canonical(self, design):
        require(isinstance(design, dict), "design must be an object")
        require(not (set(design) - set(self.fields)),
                "unknown design fields; version and complete the schema first")
        result = {}
        for name, field in self.fields.items():
            active = all(result.get(parent) in allowed
                         for parent, allowed in field.get("when", {}).items())
            value = design.get(name)
            if not active:
                require(value is None, f"inactive field must be omitted/null: {name}")
                continue
            require(name in design and value is not None,
                    f"missing active field: {name}")
            kind = field["type"]
            if kind == "number":
                require(number(value) and field["min"] <= value <= field["max"],
                        f"{name}: number outside declared bounds")
                require(type(value) is not int or float(value) == value,
                        f"{name}: integer cannot be represented exactly as a numeric descriptor")
                value = float(value)
                if value == 0:
                    value = 0.0
            elif kind == "categorical":
                require(isinstance(value, str), f"{name}: string required")
            else:
                require(isinstance(value, list) and
                        all(isinstance(x, str) for x in value),
                        f"{name}: string list required")
                value = sorted(set(value))
            choices = field.get("choices")
            if choices is not None:
                values = value if kind == "set" else [value]
                require(set(values) <= set(choices), f"{name}: unknown choice")
            result[name] = value
        require(result, "no active design fields")
        return result

    def separation(self, left, right):
        groups = {}
        terms = []
        for group in self.groups:
            distances = {}
            for field in group["fields"]:
                name = field["name"]
                a, b = name in left, name in right
                if not a and not b:
                    continue
                if a != b:
                    distance = 1.0
                elif field["type"] == "categorical":
                    distance = float(left[name] != right[name])
                elif field["type"] == "set":
                    x, y = set(left[name]), set(right[name])
                    union = x | y
                    distance = 1.0 - len(x & y) / len(union) if union else 0.0
                else:
                    x, y = left[name], right[name]
                    low, high = float(field["min"]), float(field["max"])
                    if field.get("scale", "linear") == "log":
                        x, y, low, high = map(math.log, (x, y, low, high))
                    # Scale first only when a finite bound difference overflows.
                    if not math.isfinite(high - low):
                        scale = max(abs(low), abs(high))
                        x, y, low, high = (v / scale for v in (x, y, low, high))
                    distance = min(1.0, abs(x - y) / (high - low))
                distances[name] = distance
            if distances:
                mean = sum(distances.values()) / len(distances)
                weight = float(group.get("weight", 1.0))
                groups[group["name"]] = {"distance": mean, "fields": distances}
                terms.append((weight, mean))
        require(terms, "no comparable descriptor groups")
        # Avoid overflow when a caller declares very large finite weights.
        scale = max(weight for weight, _ in terms)
        total = sum((weight / scale) * value for weight, value in terms)
        total /= sum(weight / scale for weight, _ in terms)
        return total, groups


class Record:
    def __init__(self, raw, schema):
        require(isinstance(raw, dict) and text_value(raw.get("id")),
                "record needs a nonempty id")
        require(raw.get("schema_version") == VERSION, "unsupported record version")
        require(raw.get("schema_id") == schema.raw["id"] and
                raw.get("schema_sha256") == schema.identity["sha256"],
                f"{raw['id']}: descriptor schema identity mismatch")
        require(raw.get("status") in STATUSES, f"{raw['id']}: invalid status")
        cohort = raw.get("cohort")
        require(isinstance(cohort, dict) and set(cohort) == set(COHORT_KEYS) and
                all(text_value(cohort[key]) for key in COHORT_KEYS),
                f"{raw['id']}: exact, complete cohort identity required")
        require(cohort["direction"] in {"maximize", "minimize"}, "invalid direction")
        if raw.get("score") is not None:
            require(number(raw["score"]), f"{raw['id']}: nonfinite/invalid score")
        self.raw = raw
        self.id = raw["id"]
        self.values = schema.canonical(raw.get("design"))
        self.design_id = digest({"schema": schema.identity["sha256"],
                                 "values": self.values})


def records(path, schema, identities, label):
    if path is None:
        return []
    raw, identity = load(path)
    identities[label] = identity
    require(isinstance(raw, list), f"{label}: expected a JSON array of records")
    result = [Record(item, schema) for item in raw]
    require(len({r.id for r in result}) == len(result),
            f"{label}: repeated record id")
    return result


def combine(*collections):
    by_id = {}
    for collection in collections:
        for record in collection:
            old = by_id.get(record.id)
            require(old is None or old.raw == record.raw,
                    f"conflicting snapshots for record {record.id}")
            by_id[record.id] = record
    return sorted(by_id.values(), key=lambda r: r.id)


def cohort_of(collection):
    cohorts = {digest(r.raw["cohort"]): r.raw["cohort"] for r in collection}
    require(len(cohorts) <= 1,
            "mixed evaluation cohorts: split the inputs or reevaluate first")
    return next(iter(cohorts.values()), None)


def snapshot(collection):
    return digest([r.raw for r in sorted(collection, key=lambda r: r.id)])


def novelty(record, reference, schema, k, exclude_self_design=False):
    duplicates = sorted(r.id for r in reference if r.design_id == record.design_id)
    unique = {}
    for other in sorted(reference, key=lambda r: r.id):
        if other.design_id != record.design_id:
            unique.setdefault(other.design_id, other)
    # Only retain k neighbor breakdowns, not an archive-wide distance matrix.
    nearest = []
    for other in unique.values():
        distance, groups = schema.separation(record.values, other.values)
        nearest.append({"id": other.id, "design_id": other.design_id,
                        "distance": distance, "groups": groups})
        nearest.sort(key=lambda n: (n["distance"], n["design_id"], n["id"]))
        del nearest[k:]
    bootstrap = not nearest and (exclude_self_design or not duplicates)
    if duplicates and not exclude_self_design:
        value, minimum = 0.0, 0.0
    elif nearest:
        value = sum(n["distance"] for n in nearest) / len(nearest)
        minimum = nearest[0]["distance"]
    else:
        value, minimum = 1.0, 1.0
    return {"novelty": value, "minimum_distance": minimum,
            "bootstrap_convention": bootstrap, "neighbor_count": len(nearest),
            "duplicates": duplicates, "nearest": nearest}


def score_eligibility(record):
    raw = record.raw
    if raw.get("superseded_by"):
        return "superseded_evaluation"
    if raw["status"] != "complete":
        return "not_complete"
    if raw.get("evaluation_valid") is not True:
        return "evaluation_not_validated"
    if raw.get("score_verified") is not True:
        return "saved_output_score_not_verified"
    if raw.get("aggregation_complete") is not True:
        return "required_repetitions_or_folds_not_aggregated"
    if not number(raw.get("score")):
        return "no_finite_score"
    refs = raw.get("evidence_refs")
    if not isinstance(refs, list) or not refs or not all(text_value(x) for x in refs):
        return "missing_evidence_references"
    return None


def dominates(left, right):
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right))


def fronts_of(rows):
    dominated = {r["id"]: [] for r in rows}
    counts = {r["id"]: 0 for r in rows}
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            a, b = left["objectives"], right["objectives"]
            if dominates(a, b):
                dominated[left["id"]].append(right["id"])
                counts[right["id"]] += 1
            elif dominates(b, a):
                dominated[right["id"]].append(left["id"])
                counts[left["id"]] += 1
    current = sorted(key for key, count in counts.items() if count == 0)
    fronts = []
    while current:
        fronts.append(current)
        following = []
        for key in current:
            for child in dominated[key]:
                counts[child] -= 1
                if counts[child] == 0:
                    following.append(child)
        current = sorted(following)
    return fronts


def crowd(front, by_id):
    distances = {key: 0.0 for key in front}
    boundary = set()
    for objective in range(2):
        order = sorted(front, key=lambda key: (by_id[key]["objectives"][objective], key))
        if not order:
            continue
        values = [by_id[key]["objectives"][objective] for key in order]
        if values[0] == values[-1]:
            continue
        # Normalizing before subtraction avoids overflow on opposite large scores.
        scale = max(abs(value) for value in values)
        scaled = [value / scale for value in values]
        span = scaled[-1] - scaled[0]
        boundary.update((order[0], order[-1]))
        for index in range(1, len(order) - 1):
            distances[order[index]] += (scaled[index + 1] - scaled[index - 1]) / span
    for key in front:
        by_id[key]["boundary"] = key in boundary
        by_id[key]["crowding_distance"] = None if key in boundary else distances[key]


def rank_operation(contenders, archive, schema, k, population_size):
    reference = combine(archive, contenders)
    cohort = cohort_of(reference)
    rows, excluded, designs = [], [], {}
    for record in sorted(contenders, key=lambda r: r.id):
        reason = score_eligibility(record)
        if reason:
            excluded.append({"id": record.id, "reason": reason})
            continue
        require(record.design_id not in designs,
                "multiple evaluated records for one design; provide the frozen "
                "aggregate, never the best seed/retry")
        designs[record.design_id] = record.id
        measure = novelty(record, reference, schema, k, exclude_self_design=True)
        score = float(record.raw["score"])
        quality = score if cohort["direction"] == "maximize" else -score
        rows.append({"id": record.id, "design_id": record.design_id,
                     "score": score, "objectives": [quality, measure["novelty"]],
                     "novelty_evidence": measure,
                     "verification_passed": record.raw.get("verification_passed") is True})
    by_id = {row["id"]: row for row in rows}
    fronts = fronts_of(rows)
    population = []
    for index, front in enumerate(fronts):
        crowd(front, by_id)
        for key in front:
            by_id[key]["rank"] = index
        ordered = sorted(front, key=lambda key: (
            not by_id[key]["boundary"],
            -(by_id[key]["crowding_distance"] or 0.0), key))
        population.extend(ordered[:max(0, population_size - len(population))])
    verified = [r for r in reference if score_eligibility(r) is None and
                r.raw.get("verification_passed") is True]
    require(len({r.design_id for r in verified}) == len(verified),
            "multiple verified aggregates for one design; identify the canonical "
            "aggregate and mark earlier evaluations superseded")
    sign = -1 if cohort and cohort["direction"] == "maximize" else 1
    incumbent = min(verified, key=lambda r: (sign * r.raw["score"], r.id)) if verified else None
    best = min(rows, key=lambda row: (-row["objectives"][0], row["id"])) if rows else None
    return {"cohort": cohort, "reference_snapshot": snapshot(reference),
            "fronts": fronts, "ranked": rows, "excluded": excluded,
            "selected_population_ids": population,
            "best_measured_id": best["id"] if best else None,
            "verified_incumbent_id": incumbent.id if incumbent else None,
            "verified_incumbent_score": incumbent.raw["score"] if incumbent else None,
            "summary": {"ranked_count": len(rows), "excluded_count": len(excluded),
                        "front_count": len(fronts), "active_count": len(population)}}


def to_tuple(value):
    return tuple(to_tuple(x) for x in value) if isinstance(value, list) else value


def selection_state(path, schema, identities, seed, fraction, k):
    rng = random.Random(seed)
    credit, count = 0.0, 0
    settings = {"seed": seed, "exploration_fraction": fraction, "neighbors": k}
    if path:
        state, identity = load(path)
        identities["selection_state"] = identity
        if isinstance(state, dict) and state.get("operation") == "select":
            state = state["selection_state"]
        require(isinstance(state, dict), "selection state must be an object")
        require(state.get("schema_version") == VERSION and
                state.get("schema_sha256") == schema.identity["sha256"],
                "selection state schema mismatch")
        require(state.get("settings") == settings,
                "selection settings changed; explicitly version the search policy")
        credit, count = state.get("exploration_credit"), state.get("total_admissions")
        require(number(credit) and 0 <= credit < 1, "invalid exploration credit")
        require(type(count) is int and count >= 0, "invalid admission counter")
        rng.setstate(to_tuple(state["random_state"]))
    return rng, credit, count, settings


def select_operation(args, schema, archive, candidates, pending, identities):
    reference = combine(archive, pending)
    cohort = cohort_of(combine(reference, candidates))
    archive_by_id = {r.id: r for r in archive}
    ranked, active = {}, set()
    if args.ranking:
        ranking, identity = load(args.ranking)
        identities["ranking"] = identity
        require(isinstance(ranking, dict), "ranking must be an output object")
        require(ranking.get("operation") == "rank" and
                ranking.get("schema_version") == VERSION and
                ranking.get("schema_sha256") == schema.identity["sha256"] and
                ranking.get("settings", {}).get("neighbors") == args.neighbors and
                ranking.get("calculator_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and
                ranking.get("cohort") == cohort and
                ranking.get("reference_snapshot") == snapshot(archive),
                "stale or incompatible ranking; rerank the current archive")
        ranked = {row["id"]: row for row in ranking["ranked"]}
        active = set(ranking["selected_population_ids"])
        require(active <= set(ranked) and set(ranked) <= set(archive_by_id),
                "ranking references unavailable records")
        for key, row in ranked.items():
            require(row["design_id"] == archive_by_id[key].design_id and
                    row["score"] == archive_by_id[key].raw.get("score"),
                    "ranking no longer matches archive")
    rng, credit, count, settings = selection_state(
        args.state, schema, identities, args.seed, args.exploration_fraction, args.neighbors)
    eligible, rejected = [], []
    pending_ids = {r.id for r in pending}
    for record in sorted(candidates, key=lambda r: r.id):
        raw = record.raw
        reason = None
        origin = raw.get("origin")
        parents = raw.get("parents", [])
        repetition = raw.get("repetition")
        duplicates = [r.id for r in reference if r.design_id == record.design_id]
        if record.id in archive_by_id or record.id in pending_ids:
            reason = "record_id_already_committed"
        elif raw["status"] != "proposed" or raw.get("eligible") is not True:
            reason = "not_an_eligible_proposal"
        elif origin not in ORIGINS or not text_value(raw.get("hypothesis_id")) or not text_value(raw.get("rationale")):
            reason = "missing_origin_or_supported_hypothesis"
        elif not isinstance(raw.get("evidence_refs"), list) or not raw["evidence_refs"] or not all(text_value(x) for x in raw["evidence_refs"]):
            reason = "missing_evidence_references"
        elif not isinstance(parents, list) or not all(text_value(x) for x in parents) or len(parents) != len(set(parents)):
            reason = "invalid_parent_ids"
        elif repetition is not None:
            source = archive_by_id.get(repetition.get("source_id")) if isinstance(repetition, dict) else None
            if (source is None or source.design_id != record.design_id or
                    repetition.get("kind") not in {"verification", "repetition", "retry"} or
                    not text_value(repetition.get("reason"))):
                reason = "invalid_explicit_repetition"
        elif duplicates:
            reason = "duplicate_design_requires_explicit_repetition"
        elif origin == "verification":
            reason = "verification_requires_explicit_repetition"
        if reason is None and repetition is None and origin in {"mutation", "crossover"}:
            if ((origin == "mutation" and len(parents) != 1) or
                    (origin == "crossover" and len(parents) < 2) or
                    not set(parents) <= active):
                reason = "evolutionary_parents_must_be_in_ranked_active_population"
        if reason is not None:
            rejected.append({"id": record.id, "reason": reason,
                             "duplicate_ids": sorted(duplicates)})
        else:
            eligible.append(record)
    selected = []
    while eligible and len(selected) < args.batch_size:
        measures = {r.id: novelty(r, reference, schema, args.neighbors) for r in eligible}
        tie = {r.id: rng.random() for r in sorted(eligible, key=lambda r: r.id)}
        repetitions = [r for r in eligible if r.raw.get("repetition") is not None]
        if repetitions:
            chosen = min(repetitions, key=lambda r: (tie[r.id], r.id))
            lane = "explicit_repetition"
        else:
            credit += args.exploration_fraction
            offspring = [r for r in eligible
                         if r.raw["origin"] in {"mutation", "crossover"}]
            if credit >= 1 or not offspring:
                chosen = min(eligible, key=lambda r: (
                    -measures[r.id]["minimum_distance"],
                    -measures[r.id]["novelty"], tie[r.id], r.id))
                lane = "exploration" if credit >= 1 else "exploration_fallback"
                credit = max(0.0, credit - 1.0)
            else:
                def priority(record):
                    parent = min((ranked[p] for p in record.raw["parents"]),
                                 key=lambda p: (p["rank"], not p["boundary"],
                                                -(p["crowding_distance"] or 0.0), p["id"]))
                    return (parent["rank"], not parent["boundary"],
                            -(parent["crowding_distance"] or 0.0),
                            -measures[record.id]["minimum_distance"],
                            -measures[record.id]["novelty"], tie[record.id], record.id)
                chosen = min(offspring, key=priority)
                lane = "ranked_parent_offspring"
        selected.append({"id": chosen.id, "design_id": chosen.design_id,
                         "lane": lane, "parents": chosen.raw.get("parents", []),
                         "novelty_evidence": measures[chosen.id],
                         "rationale": chosen.raw["rationale"]})
        if lane != "explicit_repetition":
            count += 1
        reference = combine(reference, [chosen])
        remaining = []
        for record in eligible:
            if record.id == chosen.id:
                continue
            if record.design_id == chosen.design_id and record.raw.get("repetition") is None:
                rejected.append({"id": record.id, "reason": "duplicate_of_selected_proposal",
                                 "duplicate_ids": [chosen.id]})
            else:
                remaining.append(record)
        eligible = remaining
    state = {"schema_version": VERSION, "schema_sha256": schema.identity["sha256"],
             "settings": settings, "exploration_credit": credit,
             "total_admissions": count, "random_state": rng.getstate()}
    return {"cohort": cohort, "reference_snapshot": snapshot(combine(archive, pending)),
            "selected": selected, "rejected": rejected,
            "unselected_ids": [r.id for r in eligible],
            "selection_state": state,
            "summary": {"selected_count": len(selected), "rejected_count": len(rejected),
                        "unselected_count": len(eligible),
                        "selected_ids": [r["id"] for r in selected]},
            "note": "Commit reservations and this state together; selection does not launch jobs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("distance", "rank", "select"):
        command = commands.add_parser(operation)
        command.add_argument("--schema", required=True)
        command.add_argument("--archive", required=True)
        command.add_argument("--neighbors", type=int, default=5)
        command.add_argument("--output", help="New JSON file; existing files are never overwritten")
        if operation == "rank":
            command.add_argument("--records", required=True)
            command.add_argument("--population-size", type=int, default=32)
        else:
            command.add_argument("--candidates", required=True)
            command.add_argument("--pending")
        if operation == "select":
            command.add_argument("--ranking")
            command.add_argument("--state")
            command.add_argument("--batch-size", type=int, required=True)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--exploration-fraction", type=float, default=0.25)
    args = parser.parse_args()
    require(args.neighbors >= 1, "neighbors must be positive")
    raw, identity = load(args.schema)
    schema = Schema(raw, identity)
    identities = {"schema": identity}
    archive = records(args.archive, schema, identities, "archive")
    if args.operation == "rank":
        require(args.population_size >= 1, "population size must be positive")
        contenders = records(args.records, schema, identities, "records")
        result = rank_operation(contenders, archive, schema, args.neighbors, args.population_size)
        settings = {"neighbors": args.neighbors, "population_size": args.population_size}
    else:
        candidates = records(args.candidates, schema, identities, "candidates")
        pending = records(args.pending, schema, identities, "pending")
        if args.operation == "select":
            require(args.batch_size >= 1, "batch size must be positive")
            require(number(args.exploration_fraction) and 0 <= args.exploration_fraction <= 1,
                    "exploration fraction must be in [0, 1]")
            result = select_operation(args, schema, archive, candidates, pending, identities)
            settings = {"neighbors": args.neighbors, "batch_size": args.batch_size,
                        "seed": args.seed, "exploration_fraction": args.exploration_fraction}
        else:
            reference = combine(archive, pending)
            cohort = cohort_of(combine(reference, candidates))
            result = {"cohort": cohort, "reference_snapshot": snapshot(reference),
                      "distances": [{"id": r.id, "design_id": r.design_id,
                                     **novelty(r, reference, schema, args.neighbors)}
                                    for r in sorted(candidates, key=lambda r: r.id)]}
            result["summary"] = {
                "candidate_count": len(candidates),
                "duplicate_candidate_count": sum(bool(r["duplicates"]) for r in result["distances"]),
                "bootstrap_count": sum(r["bootstrap_convention"] for r in result["distances"]),
            }
            settings = {"neighbors": args.neighbors}
    payload = {"schema_version": VERSION, "operation": args.operation,
               "implementation_status": "untested",
               "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "schema_id": schema.raw["id"], "schema_sha256": identity["sha256"],
               "settings": settings, "inputs": identities, **result}
    output = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(output)
    else:
        sys.stdout.write(output)
    print("Calculator is untested; outputs are selection calculations, not scientific validation.",
          file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
