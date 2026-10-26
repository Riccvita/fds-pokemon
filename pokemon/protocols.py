"""Auditable split policies, data diagnostics and execution provenance."""

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


REPEAT_SEEDS = (42, 123, 2026)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def observation_hash(battle):
    """Exact JSON observations, ignoring only top-level identifier and target.

    Object-key order is ignored; array order and numeric representation remain
    significant. This conservative criterion does not detect near duplicates.
    """
    return digest({k: v for k, v in battle.items() if k not in ("battle_id", "player_won")})


def composition_key(team):
    """Species-only unordered P1 composition, matching the existing encoder."""
    return digest(sorted(set(p["name"] for p in team)))


def connected_groups(*columns):
    """Union rows sharing any supplied key; preserve transitive duplicate links."""
    n = len(columns[0])
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for values in columns:
        seen = {}
        for i, value in enumerate(values):
            if value in seen:
                a, b = root(i), root(seen[value])
                parent[max(a, b)] = min(a, b)
            else:
                seen[value] = i
    return np.array([root(i) for i in range(n)])


def audit_battles(train, test=()):
    """Describe duplicates and recurring compositions without deleting rows."""
    ids = [str(b["battle_id"]) for b in train]
    obs = [observation_hash(b) for b in train]
    p1 = [composition_key(b["p1_team_details"]) for b in train]
    duplicate_groups = connected_groups(ids, obs)
    team_groups = connected_groups(ids, obs, p1)

    def repeats(keys):
        rows = defaultdict(list)
        for i, key in enumerate(keys):
            rows[key].append(i)
        return [{"key": str(key), "row_ids": indices,
                 "battle_ids": [ids[i] for i in indices],
                 "labels": [int(train[i]["player_won"]) for i in indices]}
                for key, indices in rows.items() if len(indices) > 1]

    counts = Counter(p1)
    distinct_observations = defaultdict(set)
    for team, fingerprint in zip(p1, obs):
        distinct_observations[team].add(fingerprint)
    test_hashes = {observation_hash(b) for b in test}
    test_ids = {str(b["battle_id"]) for b in test}
    report = {
        "n_train": len(train), "n_test": len(test),
        "duplicate_criterion": "same battle_id OR exact observations excluding battle_id/player_won; transitive closure",
        "observation_hash_rule": "sorted JSON object keys; array order preserved; no approximate matching",
        "duplicate_ids": repeats(ids), "identical_observations": repeats(obs),
        "duplicate_components": repeats(duplicate_groups),
        "conflicting_label_components": [g for g in repeats(duplicate_groups) if len(set(g["labels"])) > 1],
        "n_duplicate_components": len(set(duplicate_groups)),
        "n_p1_compositions": len(counts),
        "composition_frequency_histogram": {str(size): number for size, number in sorted(Counter(counts.values()).items())},
        "rows_with_recurring_p1_composition": sum(size for size in counts.values() if size > 1),
        "p1_compositions_with_distinct_observations": sum(len(values) > 1 for values in distinct_observations.values()),
        "train_rows_with_identical_test_observations": [i for i, h in enumerate(obs) if h in test_hashes],
        "train_rows_with_test_id": [i for i, key in enumerate(ids) if key in test_ids],
        "rows_removed": 0,
    }
    metadata = {"duplicate_groups": duplicate_groups, "team_groups": team_groups,
                "observation_hashes": obs, "p1_compositions": p1}
    return report, metadata


def split_indices(y, indices, *, seed=42, n_splits=5, groups=None):
    """Return absolute indices; preserve the original splitter when ungrouped."""
    indices = np.asarray(indices, dtype=int)
    local_y = np.asarray(y)[indices]
    if groups is None:
        if min(Counter(local_y).values()) < n_splits or len(set(local_y)) < 2:
            raise ValueError("Each class needs at least n_splits observations.")
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(indices, local_y)
    else:
        local_groups = np.asarray(groups)[indices]
        if len(set(local_groups)) < n_splits:
            raise ValueError("Not enough distinct groups for the requested folds.")
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(indices, local_y, local_groups)
    result = [(indices[fit], indices[valid]) for fit, valid in splits]
    for fit, valid in result:
        if not len(valid) or len(set(np.asarray(y)[fit])) < 2:
            raise ValueError("Split has an empty validation set or one-class training; reduce folds.")
        if groups is not None and set(np.asarray(groups)[fit]) & set(np.asarray(groups)[valid]):
            raise AssertionError("Group overlap in split")
    return result


def protocol_groups(data, split_mode):
    if split_mode not in ("random", "p1_grouped"):
        raise ValueError("split_mode must be random or p1_grouped")
    meta = data.get("audit_metadata")
    if meta is None:
        if split_mode == "p1_grouped":
            raise ValueError("Grouped validation requires raw battles from build_all.")
        return None
    if split_mode == "p1_grouped":
        return meta["team_groups"]
    groups = meta["duplicate_groups"]
    return groups if len(set(groups)) < len(groups) else None


def provenance(data):
    """Hash actual local source, including uncommitted/untracked Python code."""
    root = Path(__file__).resolve().parents[1]
    files = sorted(set(root.glob("*.py")) | set((root / "pokemon").glob("*.py"))
                   | set((root / "notebooks").glob("*.ipynb")) | set(root.glob("requirements*.txt")))
    source_hashes = {str(p.relative_to(root)): file_digest(p) for p in files}

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "code_root": str(root), "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": source_hashes, "source_bundle_sha256": digest(source_hashes),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost")},
        "installed_distributions": {dist.metadata["Name"]: dist.version
                                    for dist in importlib.metadata.distributions()},
        "input_sha256": data.get("input_sha256", {}),
        "training_records_sha256": data.get("training_records_sha256"),
    }
