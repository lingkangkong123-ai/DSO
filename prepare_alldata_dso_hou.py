import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from CoolProp import AbstractState
    import CoolProp.CoolProp as CP
except Exception:  # pragma: no cover
    AbstractState = None
    CP = None


ROOT = Path(__file__).resolve().parent
ALLDATA_DIR = ROOT / "data" / "alldata"
OUTPUT_DIR = ROOT / "outputs" / "alldata_dso"
DEFAULT_COOLPROP_PYTHON = Path.home() / ".conda" / "envs" / "dso" / "python.exe"

X_COLS = ["pi1", "pi2", "pi3", "pi4", "pi5"]
Y_COL = "m_flow_g_s"
CD_COL = "Cd_true"
DMAX_M = 1.8e-3
AREF_M2 = np.pi * (DMAX_M ** 2) / 4.0
DEFAULT_UNSEEN_REFRIGERANT = "R134a"

PKL_FOLDERS = [
    "Daten LOGIN",
    "data2",
    "newdata2",
    "newdata",
]

DIRTY_FILES = {
    "fixed_data_ihx_55_R1270.pkl",
}

REFRIGERANT_FLUID_SPECS = {
    "R1234yf": {"components": ["R1234yf"], "fractions": [1.0]},
    "R1234yf_R32_64_36": {"components": ["R1234yf", "R32"], "fractions": [0.64, 0.36]},
    "R1270": {"components": ["Propylene"], "fractions": [1.0]},
    "R1270_R600a_92_8": {"components": ["Propylene", "IsoButane"], "fractions": [0.92, 0.08]},
    "R134a": {"components": ["R134a"], "fractions": [1.0]},
    "R290": {"components": ["R290"], "fractions": [1.0]},
    "R290_R600a_63_37": {"components": ["R290", "IsoButane"], "fractions": [0.63, 0.37]},
    "R290_R600a_82_18": {"components": ["R290", "IsoButane"], "fractions": [0.82, 0.18]},
    "R600a": {"components": ["IsoButane"], "fractions": [1.0]},
}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> Optional[str]:
    normalized_map = {normalize_name(col): col for col in df.columns}
    for alias in aliases:
        col = normalized_map.get(normalize_name(alias))
        if col is not None:
            return col

    for alias in aliases:
        alias_norm = normalize_name(alias)
        for norm_col, original_col in normalized_map.items():
            if alias_norm in norm_col or norm_col in alias_norm:
                return original_col

    if required:
        raise KeyError(
            "Could not find any of these columns in the dataframe: " + ", ".join(aliases)
        )
    return None


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def expand_value_error_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    value_row = df.loc["value"]
    extracted = {}
    sample_count = None

    for col, raw_value in value_row.items():
        if raw_value is None:
            continue
        if isinstance(raw_value, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(raw_value, dtype=object).reshape(-1)
            if sample_count is None:
                sample_count = len(arr)
            elif len(arr) != sample_count:
                continue
            extracted[col] = arr

    if sample_count is None:
        raise ValueError("Could not expand value/error dataframe into sample rows.")

    return pd.DataFrame(extracted)


def normalize_loaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.index) == ["value", "error"] and "value" in df.index:
        return expand_value_error_dataframe(df)
    return df


def load_pickle_as_dataframe(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return normalize_loaded_dataframe(obj.copy())
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                return normalize_loaded_dataframe(value.copy())
        return pd.DataFrame(obj)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    raise TypeError(f"Unsupported pickle content in {path}: {type(obj)!r}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_family_from_filename(name: str) -> str:
    base = name.replace(".pkl", "")
    if base.startswith("fixed_data_simple_"):
        return base[len("fixed_data_simple_") :]
    if base.startswith("fixed_data_ihx_"):
        return base[len("fixed_data_ihx_") :]
    return base


def canonical_refrigerant_label(family: str) -> str:
    if family.startswith(("35_", "55_")):
        family = family[3:]
    if family.startswith(("B-3W", "B12W", "B2W", "B7W")) and family.endswith("_R290"):
        return "R290"
    if family == "R134a_2":
        return "R134a"
    if family.startswith("R1270_") and family not in REFRIGERANT_FLUID_SPECS:
        return "R1270"
    return family


def system_label_from_filename(name: str) -> str:
    if "fixed_data_ihx_" in name:
        return "ihx"
    if "fixed_data_simple_" in name:
        return "simple"
    return "unknown"


def safe_nanmedian_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.median(np.abs(finite)))


def safe_nanmax_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.max(np.abs(finite)))


def safe_nanmin(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.min(finite))


def maybe_rename_precomputed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    aliases = {
        "pi1": ["pi1", "Pi1"],
        "pi2": ["pi2", "Pi2"],
        "pi3": ["pi3", "Pi3"],
        "pi4": ["pi4", "Pi4"],
        "pi5": ["pi5", "Pi5", "z", "Z"],
        Y_COL: [
            "m_flow_g_s",
            "m_flow",
            "mass_flow_g_s",
            "massflow_g_s",
            "mdot_g_s",
            "m_dot_g_s",
            "m_flow_ref_gs",
        ],
    }
    for target, target_aliases in aliases.items():
        col = find_column(df, target_aliases, required=False)
        if col is not None:
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def make_coolprop_payload(rows: List[dict]) -> str:
    return json.dumps({"rows": rows, "fluid_specs": REFRIGERANT_FLUID_SPECS})


def coolprop_helper_code() -> str:
    return r"""
import json
import sys

import CoolProp.CoolProp as CP
from CoolProp import AbstractState


def build_state(spec):
    comps = spec["components"]
    fracs = spec["fractions"]
    state = AbstractState("HEOS", "&".join(comps))
    if len(comps) > 1:
        state.set_mole_fractions(fracs)
    return state


def approximate_surface_tension(spec, tsat_k, pure_cache):
    if len(spec["components"]) == 1:
        pure = pure_cache.get(spec["components"][0])
        if pure is None:
            pure = AbstractState("HEOS", spec["components"][0])
            pure_cache[spec["components"][0]] = pure
        pure.update(CP.QT_INPUTS, 0.0, tsat_k)
        return float(pure.surface_tension())

    sigma = 0.0
    for comp, frac in zip(spec["components"], spec["fractions"]):
        pure = pure_cache.get(comp)
        if pure is None:
            pure = AbstractState("HEOS", comp)
            pure_cache[comp] = pure
        pure.update(CP.QT_INPUTS, 0.0, tsat_k)
        sigma += float(frac) * float(pure.surface_tension())
    return sigma


def get_critical_props(state):
    if len(state.fluid_names()) == 1:
        return float(state.p_critical()), float(state.T_critical())
    pts = list(state.all_critical_points())
    stable = [pt for pt in pts if getattr(pt, "stable", True)]
    chosen = stable or pts
    chosen_pt = max(chosen, key=lambda pt: pt.p)
    return float(chosen_pt.p), float(chosen_pt.T)


payload = json.loads(sys.stdin.read())
fluid_specs = payload["fluid_specs"]
state_cache = {}
critical_cache = {}
pure_cache = {}
results = []

for row in payload["rows"]:
    ref = row["refrigerant"]
    spec = fluid_specs[ref]
    state = state_cache.get(ref)
    if state is None:
        state = build_state(spec)
        state_cache[ref] = state
    if ref not in critical_cache:
        critical_cache[ref] = get_critical_props(state)
    pcrit_pa, tcrit_k = critical_cache[ref]

    pin_pa = float(row["pin_pa"])
    pout_pa = float(row["pout_pa"])
    subcool_k = float(row["subcooling_k"])

    state.update(CP.PQ_INPUTS, pin_pa, 1.0)
    rho_g = float(state.rhomass())
    mu_g = float(state.viscosity())
    state.update(CP.PQ_INPUTS, pin_pa, 0.0)
    rho_f = float(state.rhomass())
    mu_f = float(state.viscosity())
    tsat_k = float(state.T())
    try:
        sigma = float(state.surface_tension())
    except Exception:
        sigma = approximate_surface_tension(spec, tsat_k, pure_cache)
    tin_k = tsat_k - subcool_k
    state.update(CP.PT_INPUTS, pin_pa, tin_k)
    rho_in = float(state.rhomass())

    nu_ratio = (mu_g / rho_g) / (mu_f / rho_f)
    sigma_over_dpin = sigma / (row["dmax_m"] * pin_pa)

    results.append(
        {
            "pcrit_pa": pcrit_pa,
            "tcrit_k": tcrit_k,
            "nu_ratio": nu_ratio,
            "sigma_over_dpin": sigma_over_dpin,
            "rho_in_kg_m3": rho_in,
            "tin_k": tin_k,
            "delta_p_pa": pin_pa - pout_pa,
        }
    )

sys.stdout.write(json.dumps(results))
"""


def compute_properties_via_subprocess(
    rows: List[dict],
    coolprop_python: Path,
) -> List[dict]:
    cmd = [str(coolprop_python), "-c", coolprop_helper_code()]
    proc = subprocess.run(
        cmd,
        input=make_coolprop_payload(rows),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "CoolProp helper subprocess failed.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def build_local_state(spec: dict):
    state = AbstractState("HEOS", "&".join(spec["components"]))
    if len(spec["components"]) > 1:
        state.set_mole_fractions(spec["fractions"])
    return state


def approximate_surface_tension_local(
    spec: dict,
    tsat_k: float,
    pure_cache: Dict[str, object],
) -> float:
    sigma = 0.0
    for comp, frac in zip(spec["components"], spec["fractions"]):
        pure = pure_cache.get(comp)
        if pure is None:
            pure = AbstractState("HEOS", comp)
            pure_cache[comp] = pure
        pure.update(CP.QT_INPUTS, 0.0, tsat_k)
        sigma += float(frac) * float(pure.surface_tension())
    return sigma


def compute_properties_locally(rows: List[dict]) -> List[dict]:
    state_cache = {}
    critical_cache = {}
    pure_cache: Dict[str, object] = {}
    results = []
    for row in rows:
        ref = row["refrigerant"]
        spec = REFRIGERANT_FLUID_SPECS[ref]
        state = state_cache.get(ref)
        if state is None:
            state = build_local_state(spec)
            state_cache[ref] = state
        if ref not in critical_cache:
            if len(spec["components"]) == 1:
                critical_cache[ref] = (float(state.p_critical()), float(state.T_critical()))
            else:
                pts = list(state.all_critical_points())
                stable = [pt for pt in pts if getattr(pt, "stable", True)]
                chosen = stable or pts
                chosen_pt = max(chosen, key=lambda pt: pt.p)
                critical_cache[ref] = (float(chosen_pt.p), float(chosen_pt.T))
        pcrit_pa, tcrit_k = critical_cache[ref]
        pin_pa = float(row["pin_pa"])
        pout_pa = float(row["pout_pa"])
        subcool_k = float(row["subcooling_k"])

        state.update(CP.PQ_INPUTS, pin_pa, 1.0)
        rho_g = float(state.rhomass())
        mu_g = float(state.viscosity())
        state.update(CP.PQ_INPUTS, pin_pa, 0.0)
        rho_f = float(state.rhomass())
        mu_f = float(state.viscosity())
        tsat_k = float(state.T())
        try:
            sigma = float(state.surface_tension())
        except Exception:
            sigma = approximate_surface_tension_local(spec, tsat_k, pure_cache)
        tin_k = tsat_k - subcool_k
        state.update(CP.PT_INPUTS, pin_pa, tin_k)
        rho_in = float(state.rhomass())

        results.append(
            {
                "pcrit_pa": pcrit_pa,
                "tcrit_k": tcrit_k,
                "nu_ratio": (mu_g / rho_g) / (mu_f / rho_f),
                "sigma_over_dpin": sigma / (row["dmax_m"] * pin_pa),
                "rho_in_kg_m3": rho_in,
                "tin_k": tin_k,
                "delta_p_pa": pin_pa - pout_pa,
            }
        )
    return results


def compute_properties(rows: List[dict], coolprop_python: Path) -> List[dict]:
    if AbstractState is not None and CP is not None:
        return compute_properties_locally(rows)
    if not coolprop_python.exists():
        raise FileNotFoundError(
            "CoolProp is not importable in the current environment and the helper "
            f"python executable was not found: {coolprop_python}"
        )
    return compute_properties_via_subprocess(rows, coolprop_python)


def scale_pressures(pin_pa: np.ndarray, pout_pa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pin_scale = safe_nanmedian_abs(pin_pa)
    pout_scale = safe_nanmedian_abs(pout_pa)
    if np.isfinite(pin_scale) and pin_scale < 1e4:
        pin_pa = pin_pa * 1e5
    if np.isfinite(pout_scale) and pout_scale < 1e4:
        pout_pa = pout_pa * 1e5
    return pin_pa, pout_pa


def normalize_opening(z: np.ndarray) -> np.ndarray:
    z_scale = safe_nanmax_abs(z)
    z_min = safe_nanmin(z)
    if np.isfinite(z_scale) and np.isfinite(z_min) and z_min >= 3.5 and z_scale > 4.0:
        return (z - 4.0) / 16.0
    return z


def pick_holdout_indices(n_rows: int, fraction: float) -> np.ndarray:
    if n_rows <= 1:
        return np.array([], dtype=int)
    count = max(1, int(np.floor(n_rows * fraction)))
    count = min(count, n_rows - 1)
    raw = np.linspace(0, n_rows - 1, count + 2)[1:-1]
    return np.unique(np.round(raw).astype(int))


def finalize_dataset(
    df: pd.DataFrame,
    *,
    source_file: str,
    source_folder: str,
    canonical_refrigerant: str,
    family: str,
    system_label: str,
    coolprop_python: Path,
) -> pd.DataFrame:
    prepared = maybe_rename_precomputed_columns(df)

    pin_col = find_column(prepared, ["p_con_out", "Pin", "p_in", "p_valve_in"])
    pout_col = find_column(prepared, ["p_eva_in", "Pout", "p_out", "p_valve_out"])
    subcool_col = find_column(
        prepared,
        ["dT_sc_con_out", "subcooling", "subcooling3c", "delta_t_uk", "dtuk"],
    )
    opening_col = find_column(
        prepared,
        ["ev_opening_simple", "opening", "valve_opening", "z", "Z"],
    )
    mflow_col = find_column(
        prepared,
        [
            Y_COL,
            "m_flow",
            "mass_flow_g_s",
            "massflow_g_s",
            "mdot_g_s",
            "m_dot_g_s",
            "m_flow_ref",
            "m_flow_ref_gs",
            "Q_flow_con_ref",
        ],
    )

    prepared = ensure_numeric(prepared, [pin_col, pout_col, subcool_col, opening_col, mflow_col])
    pin_pa = prepared[pin_col].to_numpy(dtype=float)
    pout_pa = prepared[pout_col].to_numpy(dtype=float)
    pin_pa, pout_pa = scale_pressures(pin_pa, pout_pa)
    subcool_k = prepared[subcool_col].to_numpy(dtype=float)
    z = normalize_opening(prepared[opening_col].to_numpy(dtype=float))
    m_flow_g_s = prepared[mflow_col].to_numpy(dtype=float)

    requests = []
    for pin_value, pout_value, subcool_value in zip(pin_pa, pout_pa, subcool_k):
        requests.append(
            {
                "refrigerant": canonical_refrigerant,
                "pin_pa": float(pin_value),
                "pout_pa": float(pout_value),
                "subcooling_k": float(subcool_value),
                "dmax_m": DMAX_M,
            }
        )
    properties = compute_properties(requests, coolprop_python)

    pcrit_pa = np.array([row["pcrit_pa"] for row in properties], dtype=float)
    tcrit_k = np.array([row["tcrit_k"] for row in properties], dtype=float)
    nu_ratio = np.array([row["nu_ratio"] for row in properties], dtype=float)
    sigma_over_dpin = np.array([row["sigma_over_dpin"] for row in properties], dtype=float)
    rho_in = np.array([row["rho_in_kg_m3"] for row in properties], dtype=float)
    tin_k = np.array([row["tin_k"] for row in properties], dtype=float)
    delta_p_pa = np.array([row["delta_p_pa"] for row in properties], dtype=float)

    pi1 = np.divide(pin_pa - pout_pa, pcrit_pa, out=np.full_like(pin_pa, np.nan), where=np.abs(pcrit_pa) > 0.0)
    pi2 = np.divide(subcool_k, tcrit_k, out=np.full_like(subcool_k, np.nan), where=np.abs(tcrit_k) > 0.0)
    pi3 = nu_ratio
    pi4 = sigma_over_dpin
    pi5 = z

    m_flow_kg_s = m_flow_g_s * 1e-3
    flow_factor = AREF_M2 * np.sqrt(np.maximum(2.0 * rho_in * delta_p_pa, 0.0))
    cd_true = np.divide(
        m_flow_kg_s,
        flow_factor,
        out=np.full_like(m_flow_kg_s, np.nan),
        where=np.abs(flow_factor) > 1e-12,
    )

    out = pd.DataFrame(
        {
            "source_folder": source_folder,
            "source_file": source_file,
            "family": family,
            "system_label": system_label,
            "refrigerant": canonical_refrigerant,
            "pi1": pi1,
            "pi2": pi2,
            "pi3": pi3,
            "pi4": pi4,
            "pi5": pi5,
            Y_COL: m_flow_g_s,
            "m_flow_kg_s": m_flow_kg_s,
            "Pin_Pa": pin_pa,
            "Pout_Pa": pout_pa,
            "deltaP_Pa": delta_p_pa,
            "subcooling_K": subcool_k,
            "Tin_K": tin_k,
            "rho_in_kg_m3": rho_in,
            "pcrit_Pa": pcrit_pa,
            "tcrit_K": tcrit_k,
            "Aref_m2": AREF_M2,
            CD_COL: cd_true,
        }
    )

    finite_mask = np.isfinite(out[X_COLS + [Y_COL, "rho_in_kg_m3", "deltaP_Pa", CD_COL]].to_numpy(dtype=float)).all(axis=1)
    positive_mask = (out["rho_in_kg_m3"] > 0.0) & (out["deltaP_Pa"] > 0.0)
    out = out.loc[finite_mask & positive_mask].reset_index(drop=True)
    return out


def collect_unique_pickle_files() -> List[dict]:
    records = []
    for folder_name in PKL_FOLDERS:
        folder = ALLDATA_DIR / folder_name
        for path in sorted(folder.glob("*.pkl")):
            family = extract_family_from_filename(path.name)
            canonical_ref = canonical_refrigerant_label(family)
            if canonical_ref not in REFRIGERANT_FLUID_SPECS:
                raise KeyError(
                    f"No refrigerant fluid mapping configured for {canonical_ref} from {path.name}"
                )
            records.append(
                {
                    "path": path,
                    "folder": folder_name,
                    "file": path.name,
                    "family": family,
                    "refrigerant": canonical_ref,
                    "system_label": system_label_from_filename(path.name),
                    "sha256": sha256_file(path),
                }
            )

    deduped = {}
    for record in records:
        key = record["sha256"]
        if key not in deduped:
            deduped[key] = record
    return sorted(deduped.values(), key=lambda row: (row["refrigerant"], row["family"], row["file"]))


def build_all_frames(coolprop_python: Path) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    unique_records = collect_unique_pickle_files()
    skipped_records = []
    processed_frames = []

    for record in unique_records:
        if record["file"] in DIRTY_FILES:
            skipped_records.append(
                {
                    "file": record["file"],
                    "reason": "known_dirty_from_readme",
                    "folder": record["folder"],
                    "refrigerant": record["refrigerant"],
                }
            )
            continue

        raw_df = load_pickle_as_dataframe(record["path"])
        prepared_df = finalize_dataset(
            raw_df,
            source_file=record["file"],
            source_folder=record["folder"],
            canonical_refrigerant=record["refrigerant"],
            family=record["family"],
            system_label=record["system_label"],
            coolprop_python=coolprop_python,
        )
        prepared_df["dataset_sha256"] = record["sha256"]
        processed_frames.append(prepared_df)

    all_df = pd.concat(processed_frames, ignore_index=True)
    manifest = []
    for record in unique_records:
        if record["file"] in DIRTY_FILES:
            continue
        rows = int((all_df["dataset_sha256"] == record["sha256"]).sum())
        manifest.append(
            {
                "file": record["file"],
                "folder": record["folder"],
                "family": record["family"],
                "refrigerant": record["refrigerant"],
                "system_label": record["system_label"],
                "dataset_sha256": record["sha256"],
                "rows_after_cleaning": rows,
            }
        )
    return all_df, manifest, skipped_records


def build_splits(
    all_df: pd.DataFrame,
    *,
    unseen_refrigerant: str,
    seen_test_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    unseen_mask = all_df["refrigerant"] == unseen_refrigerant
    df_unseen = all_df.loc[unseen_mask].copy().reset_index(drop=True)
    seen_df = all_df.loc[~unseen_mask].copy()

    train_parts = []
    seen_test_parts = []
    seen_manifest = []

    for source_file, group in seen_df.groupby("source_file", sort=True):
        group = group.sort_values(["pi5", "m_flow_g_s", "pi1"]).reset_index(drop=True)
        holdout_idx = pick_holdout_indices(len(group), seen_test_fraction)
        seen_mask = np.zeros(len(group), dtype=bool)
        seen_mask[holdout_idx] = True
        train_part = group.loc[~seen_mask].copy()
        seen_test_part = group.loc[seen_mask].copy()
        train_parts.append(train_part)
        if not seen_test_part.empty:
            seen_test_parts.append(seen_test_part)
        seen_manifest.append(
            {
                "source_file": source_file,
                "refrigerant": str(group["refrigerant"].iloc[0]),
                "rows_total": int(len(group)),
                "rows_train": int(len(train_part)),
                "rows_seen_test": int(len(seen_test_part)),
            }
        )

    df_train = pd.concat(train_parts, ignore_index=True)
    df_seen_test = pd.concat(seen_test_parts, ignore_index=True)

    split_summary = {
        "unseen_refrigerant": unseen_refrigerant,
        "seen_test_fraction": seen_test_fraction,
        "train_samples": int(len(df_train)),
        "seen_test_samples": int(len(df_seen_test)),
        "unseen_test_samples": int(len(df_unseen)),
        "train_refrigerants": sorted(df_train["refrigerant"].unique().tolist()),
        "seen_test_refrigerants": sorted(df_seen_test["refrigerant"].unique().tolist()),
        "unseen_test_refrigerants": sorted(df_unseen["refrigerant"].unique().tolist()),
        "seen_split_manifest": seen_manifest,
    }
    return df_train, df_seen_test, df_unseen, split_summary


def export_regression_csv(df: pd.DataFrame, path: Path, target_col: str, with_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[X_COLS + [target_col]].to_csv(path, index=False, header=with_header)


def write_outputs(
    df_all: pd.DataFrame,
    df_train: pd.DataFrame,
    df_seen_test: pd.DataFrame,
    df_unseen_test: pd.DataFrame,
    manifest: List[dict],
    skipped_records: List[dict],
    split_summary: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_regression_csv(df_train, output_dir / "train_cd_dso_no_header.csv", CD_COL, with_header=False)
    export_regression_csv(df_train, output_dir / "train_cd_dso_with_header.csv", CD_COL, with_header=True)
    export_regression_csv(df_train, output_dir / "train_mflow_dso_no_header.csv", Y_COL, with_header=False)
    export_regression_csv(df_train, output_dir / "train_mflow_dso_with_header.csv", Y_COL, with_header=True)

    df_all.to_csv(output_dir / "all_processed.csv", index=False)
    df_train.to_csv(output_dir / "train_processed.csv", index=False)
    df_seen_test.to_csv(output_dir / "seen_test_processed.csv", index=False)
    df_unseen_test.to_csv(output_dir / "unseen_test_processed.csv", index=False)

    summary = {
        "all_samples": int(len(df_all)),
        "independent_dataset_count": int(df_all["dataset_sha256"].nunique()),
        "dirty_files_skipped": skipped_records,
        "inputs": X_COLS,
        "cd_output": CD_COL,
        "mass_flow_output": Y_COL,
        "dmax_m": DMAX_M,
        "aref_m2": AREF_M2,
        "manifest": manifest,
        "split_summary": split_summary,
    }

    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare alldata refrigerant datasets for Cd-first symbolic fitting and "
            "mass-flow back-calculation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where processed CSV and JSON files will be written.",
    )
    parser.add_argument(
        "--unseen-refrigerant",
        type=str,
        default=DEFAULT_UNSEEN_REFRIGERANT,
        help="Canonical refrigerant label to hold out entirely for unseen testing.",
    )
    parser.add_argument(
        "--seen-test-fraction",
        type=float,
        default=0.25,
        help="Per-source-file fraction of seen-refrigerant rows reserved for seen_test.",
    )
    parser.add_argument(
        "--coolprop-python",
        type=Path,
        default=DEFAULT_COOLPROP_PYTHON,
        help="Python executable that has CoolProp available, used when this environment lacks it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_all, manifest, skipped_records = build_all_frames(args.coolprop_python)
    df_train, df_seen_test, df_unseen_test, split_summary = build_splits(
        df_all,
        unseen_refrigerant=args.unseen_refrigerant,
        seen_test_fraction=args.seen_test_fraction,
    )
    write_outputs(
        df_all,
        df_train,
        df_seen_test,
        df_unseen_test,
        manifest,
        skipped_records,
        split_summary,
        args.output_dir,
    )

    print(f"All samples: {len(df_all)}")
    print(f"Train samples: {len(df_train)}")
    print(f"Seen-test samples: {len(df_seen_test)}")
    print(f"Unseen-test samples: {len(df_unseen_test)}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
