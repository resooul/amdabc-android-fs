"""Stage 0: clean data, outer folds and scenario splits.

Input: per-source CSVs ``{data_dir}/{rep} {source}.csv`` (columns: uygulama_adi,
opcode presence columns, Class with 1 = malware). These files reproduce the
merged dvm/jvm/hybrid tables exactly (checked when the merged file is present)
and carry the source label needed for the scenarios.

Cleaning, per representation and in this order (counts go to the data card):
 1. drop the identifier column ``uygulama_adi``;
 2. drop ``<name>.1`` columns that are exact copies of ``<name>``;
 3. drop constant columns;
 4. drop rows with no opcode at all (all-zero vectors);
 5. drop vectors that occur with both labels;
 6. collapse identical (vector, label) rows into one row, keeping the set of
    sources it came from and its multiplicity;
 7. drop columns that are constant, or identical to an earlier column, on the
    cleaned rows (aliases are listed in the data card).
Genome is merged into Drebin as one malware source (all Genome samples are in
Drebin).

Outputs (results/data/): {rep}.npz, {rep}_folds.csv, {rep}_scenarios.json,
data_card.json.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..io import read_json, write_csv, write_json, write_npz

log = logging.getLogger("amdabc")
ID_COL, LABEL = "uygulama_adi", "Class"
MERGE_SOURCE = {"Genome": "Drebin"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rowhash(df: pd.DataFrame) -> np.ndarray:
    return np.sort(pd.util.hash_pandas_object(df, index=False).to_numpy())


def load_sources(cfg: dict, rep: str):
    ddir = Path(cfg["data_dir"])
    srcs = cfg["sources"]["malware"] + cfg["sources"]["benign"]
    parts, hashes = [], {}
    for s in srcs:
        p = ddir / f"{rep} {s}.csv"
        hashes[p.name] = sha256(p)
        df = pd.read_csv(p)
        parts.append(pd.concat([df, pd.Series(MERGE_SOURCE.get(s, s), index=df.index, name="__source")], axis=1))
    df = pd.concat(parts, ignore_index=True)
    merged = ddir / f"{rep}.csv"
    if merged.exists():
        m = pd.read_csv(merged)
        hashes[merged.name] = sha256(merged)
        same = len(m) == len(df) and np.array_equal(_rowhash(m), _rowhash(df[m.columns]))
        if not same:
            raise ValueError(f"{rep}: per-source files do not reproduce {merged.name}")
    return df, hashes


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    card = {"rows_raw": len(df), "cols_raw": df.shape[1] - 1}   # minus helper column
    df = df.drop(columns=[ID_COL])
    feats = [c for c in df.columns if c not in (LABEL, "__source")]

    dup = [c for c in feats if c.endswith(".1") and c[:-2] in df.columns and df[c].equals(df[c[:-2]])]
    feats = [c for c in feats if c not in dup]
    const = [c for c in feats if df[c].nunique() <= 1]
    feats = [c for c in feats if c not in const]
    card.update(dup_cols_dropped=len(dup), const_cols_dropped=len(const), n_features=len(feats))

    X = df[feats].to_numpy(np.uint8)
    zero = X.sum(1) == 0
    card["zero_rows_dropped"] = int(zero.sum())
    card["zero_rows_by_source"] = df.loc[zero, "__source"].value_counts().to_dict()
    df = df.loc[~zero].reset_index(drop=True)

    key = pd.util.hash_pandas_object(df[feats], index=False)
    labels_per_vec = df.groupby(key.values)[LABEL].nunique()
    conflict_keys = set(labels_per_vec.index[labels_per_vec > 1])
    conflict = key.isin(conflict_keys).to_numpy()
    card["conflict_vectors"] = len(conflict_keys)
    card["conflict_rows_dropped"] = int(conflict.sum())
    df = df.loc[~conflict].reset_index(drop=True)

    key = pd.util.hash_pandas_object(df[feats + [LABEL]], index=False).to_numpy()
    df = df.copy()
    df["__key"] = key
    g = df.groupby("__key", sort=False)
    first = g.head(1).copy()
    srcsets = g["__source"].agg(lambda s: "|".join(sorted(set(s))))
    counts = g.size()
    first["__sources"] = first["__key"].map(srcsets)
    first["__count"] = first["__key"].map(counts)
    first = first.reset_index(drop=True)
    card["duplicate_rows_collapsed"] = int(len(df) - len(first))

    # 7. columns that became constant, or identical in content to an earlier column,
    #    on the cleaned rows (lossless: identical columns carry the same information)
    Xc = first[feats].to_numpy(np.uint8)
    const2 = Xc.min(0) == Xc.max(0)
    _, first_idx = np.unique(Xc.T, axis=0, return_index=True)
    keep = np.zeros(len(feats), bool)
    keep[first_idx] = True
    keep &= ~const2
    aliases = {}
    for j in np.where(~keep & ~const2)[0]:
        k = int(np.where((Xc.T == Xc[:, j]).all(1) & keep)[0][0])
        aliases.setdefault(feats[k], []).append(feats[j])
    card["const_cols_after_cleaning"] = int(const2.sum())
    card["identical_cols_dropped"] = int((~keep & ~const2).sum())
    card["identical_col_aliases"] = aliases
    feats = [f for f, k in zip(feats, keep) if k]
    card["n_features"] = len(feats)
    card["rows_clean"] = len(first)
    card["malware"] = int((first[LABEL] == 1).sum())
    card["benign"] = int((first[LABEL] == 0).sum())
    return first[feats + [LABEL, "__sources", "__count"]], card


def scenario_split(sources: np.ndarray, y: np.ndarray, sdef: dict, test_frac: float, seed: int) -> dict:
    srcsets = [set(s.split("|")) for s in sources]
    test_b, train_b = set(sdef["test_benign"]), set(sdef["train_benign"])
    ben = np.where(y == 0)[0]
    test_ben = [i for i in ben if srcsets[i] & test_b]
    train_ben = [i for i in ben if not (srcsets[i] & test_b) and srcsets[i] <= train_b]
    mal = np.where(y == 1)[0]
    mal_tr, mal_te = train_test_split(mal, test_size=test_frac, random_state=seed)
    return {"train": sorted(map(int, list(train_ben) + list(mal_tr))),
            "test": sorted(map(int, list(test_ben) + list(mal_te)))}


def prepare(cfg: dict) -> None:
    out = Path(cfg["results_dir"]) / "data"
    card_path = out / "data_card.json"
    if card_path.exists():
        log.info("prepare: done (%s exists)", card_path)
        return
    card = {}
    for rep in cfg["representations"]:
        df, hashes = load_sources(cfg, rep)
        clean_df, c = clean(df)
        feats = [x for x in clean_df.columns if not x.startswith("__") and x != LABEL]
        X = clean_df[feats].to_numpy(np.uint8)
        y = clean_df[LABEL].to_numpy(np.int8)
        sources = clean_df["__sources"].to_numpy(str)
        write_npz(out / f"{rep}.npz", X=X, y=y, features=np.array(feats), sources=sources,
                  count=clean_df["__count"].to_numpy(np.int32))
        skf = StratifiedKFold(cfg["split"]["n_folds"], shuffle=True, random_state=cfg["split"]["seed"])
        fold = np.empty(len(y), np.int8)
        for k, (_, te) in enumerate(skf.split(X, y)):
            fold[te] = k
        write_csv(out / f"{rep}_folds.csv", pd.DataFrame({"row": np.arange(len(y)), "fold": fold}), index=False)
        sc = {name: scenario_split(sources, y, cfg["scenarios"]["definitions"][name],
                                   cfg["scenarios"]["malware_test_fraction"], cfg["split"]["seed"])
              for name in cfg["scenarios"]["names"]}
        write_json(out / f"{rep}_scenarios.json", sc)
        c["scenarios"] = {k: {"train": len(v["train"]), "test": len(v["test"])} for k, v in sc.items()}
        c["input_sha256"] = hashes
        c["output_sha256"] = sha256(out / f"{rep}.npz")
        card[rep] = c
        log.info("prepare %s: %d -> %d rows, %d features", rep, c["rows_raw"], c["rows_clean"], c["n_features"])
    write_json(card_path, card)


# ---- accessors used by later stages -------------------------------------------
def load_rep(cfg: dict, rep: str) -> dict:
    out = Path(cfg["results_dir"]) / "data"
    with np.load(out / f"{rep}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["fold"] = pd.read_csv(out / f"{rep}_folds.csv")["fold"].to_numpy()
    return d


def outer_split(d: dict, k: int):
    tr = np.where(d["fold"] != k)[0]
    te = np.where(d["fold"] == k)[0]
    return tr, te


def scenario_indices(cfg: dict, rep: str, name: str):
    sc = read_json(Path(cfg["results_dir"]) / "data" / f"{rep}_scenarios.json")[name]
    return np.array(sc["train"]), np.array(sc["test"])
