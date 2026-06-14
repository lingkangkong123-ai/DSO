import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pickle

try:
    from CoolProp.CoolProp import PropsSI
except Exception:  # pragma: no cover - script can still work if pi columns already exist
    PropsSI = None

try:
    import pickle5  # type: ignore
except Exception:  # pragma: no cover - optional compatibility dependency
    pickle5 = None


ROOT = Path(__file__).resolve().parent
NEW_DATA_DIR = ROOT / "new_data"
OUTPUT_DIR = ROOT / "outputs" / "new_data_dso"

TEST_FILES = {
    "fixed_data_simple_B-3W50_R290.pkl",
    "fixed_data_simple_B7W35_R290.pkl",
}

TRAIN_FILES = {
    "fixed_data_simple_B-3W35_R290.pkl",
    "fixed_data_simple_B-3W65_R290.pkl",
    "fixed_data_simple_B12W35_R290.pkl",
    "fixed_data_simple_B12W50_R290.pkl",
    "fixed_data_simple_B12W65_R290.pkl",
    "fixed_data_simple_B2W35_R290.pkl",
    "fixed_data_simple_B2W50_R290.pkl",
    "fixed_data_simple_B2W65_R290.pkl",
    "fixed_data_simple_B7W50_R290.pkl",
    "fixed_data_simple_B7W65_R290.pkl",
}

X_COLS = ["pi1", "pi2", "pi3", "pi4", "pi5"]
Y_COL = "m_flow_g_s"
REFRIGERANT = "Propane"
DREF = 1.8e-3
AREF = np.pi * (DREF ** 2) / 4.0

PCRIT = PropsSI(REFRIGERANT, "pcrit") if PropsSI is not None else None
TCRIT_C = PropsSI(REFRIGERANT, "Tcrit") - 273.15 if PropsSI is not None else None


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
            "Could not find any of these columns in the dataframe: "
            + ", ".join(aliases)
        )
    return None


def load_pickle_as_dataframe(path: Path) -> pd.DataFrame:
    try:
        obj = pd.read_pickle(path)
    except ValueError as exc:
        message = str(exc)
        if "unsupported pickle protocol: 5" not in message:
            raise

        if pickle5 is None:
            raise RuntimeError(
                f"{path.name} was saved with pickle protocol 5. "
                "Your current Python is too old to read it directly. "
                "Please use Python 3.8+ or install `pickle5` into this environment "
                "and rerun the script."
            ) from exc

        with open(path, "rb") as f:
            try:
                obj = pickle5.load(f)
            except AttributeError as attr_exc:
                if "_unpickle_block" not in str(attr_exc):
                    raise
                raise RuntimeError(
                    f"{path.name} depends on a newer pandas pickle format than the one in "
                    "your current environment. This usually means the file was created with "
                    "a newer pandas/Python stack, while your `dso` environment is older. "
                    "Please read/convert these pickle files in a newer environment "
                    "(recommended: Python 3.10+ with a recent pandas), export them to CSV, "
                    "and then use the generated CSV files in the DSO environment."
                ) from attr_exc
    except AttributeError as exc:
        if "_unpickle_block" not in str(exc):
            raise
        raise RuntimeError(
            f"{path.name} depends on a newer pandas pickle format than the one in "
            "your current environment. This usually means the file was created with "
            "a newer pandas/Python stack, while your `dso` environment is older. "
            "Please read/convert these pickle files in a newer environment "
            "(recommended: Python 3.10+ with a recent pandas), export them to CSV, "
            "and then use the generated CSV files in the DSO environment."
        ) from exc

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


def normalize_loaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.index) == ["value", "error"] and "value" in df.index:
        return expand_value_error_dataframe(df)
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


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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

    precomputed_aliases = {
        "pi1": ["pi1", "Pi1"],
        "pi2": ["pi2", "Pi2"],
        "pi3": ["pi3", "Pi3"],
        "pi4": ["pi4", "Pi4"],
        "pi5": ["pi5", "Pi5", "z", "Z"],
        "m_flow_g_s": [
            "m_flow_g_s",
            "m_flow",
            "mass_flow_g_s",
            "massflow_g_s",
            "mdot_g_s",
            "m_dot_g_s",
            "m_flow_ref",
            "m_flow_ref_gs",
            "Q_flow_con_ref",
        ],
    }

    for target, aliases in precomputed_aliases.items():
        col = find_column(df, aliases, required=False)
        if col is not None:
            rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def calculate_pi_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = maybe_rename_precomputed_columns(df)

    if all(col in df.columns for col in X_COLS + [Y_COL]):
        return ensure_numeric(df, X_COLS + [Y_COL])

    if PropsSI is None:
        raise RuntimeError(
            "CoolProp is required when the pickle files do not already contain "
            "pi1-pi5 and m_flow_g_s columns."
        )

    pin_col = find_column(df, ["p_con_out", "Pin", "p_in", "p_valve_in"])
    pout_col = find_column(df, ["p_eva_in", "Pout", "p_out", "p_valve_out"])
    subcool_col = find_column(
        df,
        ["dT_sc_con_out", "subcooling", "subcooling3c", "delta_t_uk", "dtuk"],
    )
    opening_col = find_column(
        df,
        ["ev_opening_simple", "opening", "valve_opening", "z", "Z"],
    )
    mflow_col = find_column(
        df,
        [
            "m_flow_g_s",
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

    df = df.copy()
    df = ensure_numeric(df, [pin_col, pout_col, subcool_col, opening_col, mflow_col])

    pin_pa = df[pin_col].to_numpy(dtype=float)
    pout_pa = df[pout_col].to_numpy(dtype=float)

    # Stored pressures are often in bar in this project; convert only when values look small.
    pin_scale = safe_nanmedian_abs(pin_pa)
    pout_scale = safe_nanmedian_abs(pout_pa)
    if np.isfinite(pin_scale) and pin_scale < 1e4:
        pin_pa = pin_pa * 1e5
    if np.isfinite(pout_scale) and pout_scale < 1e4:
        pout_pa = pout_pa * 1e5

    z = df[opening_col].to_numpy(dtype=float)
    z_scale = safe_nanmax_abs(z)
    z_min = safe_nanmin(z)
    if np.isfinite(z_scale) and np.isfinite(z_min) and z_min >= 3.5 and z_scale > 4.0:
        z = (z - 4.0) / 16.0

    df["pi1"] = (pin_pa - pout_pa) / PCRIT
    df["pi2"] = df[subcool_col].to_numpy(dtype=float) / TCRIT_C
    df["pi3"] = np.nan
    df["pi4"] = np.nan
    df["pi5"] = z
    df["m_flow_g_s"] = df[mflow_col].to_numpy(dtype=float)

    if "nu_g" in df.columns and "nu_f" in df.columns:
        df = ensure_numeric(df, ["nu_g", "nu_f"])
        df["pi3"] = df["nu_g"] / df["nu_f"]
    else:
        df["pi3"] = [
            safe_nu_ratio(pin_value)
            for pin_value in pin_pa
        ]

    if "sigma" in df.columns:
        df = ensure_numeric(df, ["sigma"])
        df["pi4"] = df["sigma"] / (DREF * pin_pa)
    else:
        df["pi4"] = [
            safe_sigma_over_dp(pin_value)
            for pin_value in pin_pa
        ]

    return df


def safe_nu_ratio(pin_pa: float) -> float:
    try:
        density_g = PropsSI("D", "P", pin_pa, "Q", 1, REFRIGERANT)
        viscosity_g = PropsSI("V", "P", pin_pa, "Q", 1, REFRIGERANT)
        density_f = PropsSI("D", "P", pin_pa, "Q", 0, REFRIGERANT)
        viscosity_f = PropsSI("V", "P", pin_pa, "Q", 0, REFRIGERANT)
        return (viscosity_g / density_g) / (viscosity_f / density_f)
    except Exception:
        return np.nan


def safe_sigma_over_dp(pin_pa: float) -> float:
    try:
        sigma = PropsSI("I", "P", pin_pa, "Q", 0, REFRIGERANT)
        return sigma / (DREF * pin_pa)
    except Exception:
        return np.nan


def safe_tsat_k(pin_pa: float) -> float:
    try:
        return PropsSI("T", "P", pin_pa, "Q", 0, REFRIGERANT)
    except Exception:
        return np.nan


def safe_density_tp(t_k: float, pin_pa: float) -> float:
    try:
        return PropsSI("D", "T", t_k, "P", pin_pa, REFRIGERANT)
    except Exception:
        return np.nan


def finalize_dataset(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    prepared = calculate_pi_groups(df)
    pin_col = find_column(prepared, ["p_con_out", "Pin", "p_in", "p_valve_in"])
    pout_col = find_column(prepared, ["p_eva_in", "Pout", "p_out", "p_valve_out"])
    subcool_col = find_column(
        prepared,
        ["dT_sc_con_out", "subcooling", "subcooling3c", "delta_t_uk", "dtuk"],
    )

    prepared = ensure_numeric(prepared, [pin_col, pout_col, subcool_col])
    pin_pa = prepared[pin_col].to_numpy(dtype=float)
    pout_pa = prepared[pout_col].to_numpy(dtype=float)
    subcool_k = prepared[subcool_col].to_numpy(dtype=float)

    pin_scale = safe_nanmedian_abs(pin_pa)
    pout_scale = safe_nanmedian_abs(pout_pa)
    if np.isfinite(pin_scale) and pin_scale < 1e4:
        pin_pa = pin_pa * 1e5
    if np.isfinite(pout_scale) and pout_scale < 1e4:
        pout_pa = pout_pa * 1e5

    tsat_k = np.array([safe_tsat_k(value) for value in pin_pa], dtype=float)
    tin_k = tsat_k - subcool_k
    rho_in = np.array(
        [safe_density_tp(t_value, p_value) for t_value, p_value in zip(tin_k, pin_pa)],
        dtype=float,
    )

    m_flow_g_s = prepared[Y_COL].to_numpy(dtype=float)
    m_flow_kg_s = m_flow_g_s * 1e-3
    delta_p_pa = pin_pa - pout_pa
    flow_factor = AREF * np.sqrt(np.maximum(2.0 * rho_in * delta_p_pa, 0.0))
    cd_true = np.divide(
        m_flow_kg_s,
        flow_factor,
        out=np.full_like(m_flow_kg_s, np.nan, dtype=float),
        where=np.abs(flow_factor) > 1e-12,
    )

    prepared["Pin_Pa"] = pin_pa
    prepared["Pout_Pa"] = pout_pa
    prepared["deltaP_Pa"] = delta_p_pa
    prepared["subcooling_K"] = subcool_k
    prepared["Tin_K"] = tin_k
    prepared["rho_in_kg_m3"] = rho_in
    prepared["m_flow_kg_s"] = m_flow_kg_s
    prepared["Aref_m2"] = AREF
    prepared["Cd_true"] = cd_true
    prepared["source_file"] = source_file
    prepared = ensure_numeric(
        prepared,
        X_COLS + [Y_COL, "Pin_Pa", "Pout_Pa", "deltaP_Pa", "Tin_K", "rho_in_kg_m3", "Cd_true"],
    )

    required_cols = X_COLS + [Y_COL, "Pin_Pa", "Pout_Pa", "deltaP_Pa", "Tin_K", "rho_in_kg_m3", "Cd_true"]
    missing = prepared[required_cols].isna().any(axis=1)
    if missing.any():
        missing_cols = prepared.loc[missing, required_cols].isna().sum()
        raise ValueError(
            f"{source_file} still has missing required values after processing: "
            f"{missing_cols[missing_cols > 0].to_dict()}"
        )

    keep_cols = [
        "source_file",
        *X_COLS,
        Y_COL,
        "m_flow_kg_s",
        "Pin_Pa",
        "Pout_Pa",
        "deltaP_Pa",
        "subcooling_K",
        "Tin_K",
        "rho_in_kg_m3",
        "Aref_m2",
        "Cd_true",
    ]
    return prepared[keep_cols].reset_index(drop=True)


def export_dso_csv(df: pd.DataFrame, path: Path, with_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[X_COLS + [Y_COL]].to_csv(path, index=False, header=with_header)


def export_cd_dso_csv(df: pd.DataFrame, path: Path, with_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[X_COLS + ["Cd_true"]].to_csv(path, index=False, header=with_header)


def validate_split(files_on_disk: Sequence[Path]) -> None:
    names_on_disk = {path.name for path in files_on_disk}
    expected = TRAIN_FILES | TEST_FILES

    missing = sorted(expected - names_on_disk)
    extra = sorted(names_on_disk - expected)

    if missing:
        raise FileNotFoundError(f"Missing expected pickle files: {missing}")
    if extra:
        print(f"Warning: extra pickle files found and ignored: {extra}")


def build_split_frames(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    files_on_disk = sorted(input_dir.glob("*.pkl"))
    validate_split(files_on_disk)

    train_frames = []
    test_frames = []
    manifest = []

    for path in files_on_disk:
        if path.name not in TRAIN_FILES and path.name not in TEST_FILES:
            continue

        split = "test" if path.name in TEST_FILES else "train"
        raw_df = load_pickle_as_dataframe(path)
        prepared_df = finalize_dataset(raw_df, path.name)

        manifest.append(
            {
                "file": path.name,
                "split": split,
                "rows": int(len(prepared_df)),
                "columns": X_COLS + [Y_COL],
            }
        )

        if split == "train":
            train_frames.append(prepared_df)
        else:
            test_frames.append(prepared_df)

    df_train = pd.concat(train_frames, ignore_index=True)
    df_test = pd.concat(test_frames, ignore_index=True)
    return df_train, df_test, manifest


def write_outputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    manifest: List[dict],
    input_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_dso_csv(df_train, output_dir / "train_dso_no_header.csv", with_header=False)
    export_dso_csv(df_test, output_dir / "test_dso_no_header.csv", with_header=False)
    export_dso_csv(df_train, output_dir / "train_dso_with_header.csv", with_header=True)
    export_dso_csv(df_test, output_dir / "test_dso_with_header.csv", with_header=True)
    export_cd_dso_csv(df_train, output_dir / "train_cd_dso_no_header.csv", with_header=False)
    export_cd_dso_csv(df_test, output_dir / "test_cd_dso_no_header.csv", with_header=False)
    export_cd_dso_csv(df_train, output_dir / "train_cd_dso_with_header.csv", with_header=True)
    export_cd_dso_csv(df_test, output_dir / "test_cd_dso_with_header.csv", with_header=True)

    df_train.to_csv(output_dir / "train_processed.csv", index=False)
    df_test.to_csv(output_dir / "test_processed.csv", index=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_files": sorted(TRAIN_FILES),
        "test_files": sorted(TEST_FILES),
        "train_samples": int(len(df_train)),
        "test_samples": int(len(df_test)),
        "inputs": X_COLS,
        "output": Y_COL,
        "cd_output": "Cd_true",
        "aref_m2": AREF,
        "manifest": manifest,
    }

    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split new_data pickle files into the requested train/test groups and "
            "export DSO-ready CSV files using pi1-pi5 as inputs and m_flow_g_s as target."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=NEW_DATA_DIR,
        help="Directory containing the 12 pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where DSO-ready files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_train, df_test, manifest = build_split_frames(args.input_dir)
    write_outputs(df_train, df_test, manifest, args.input_dir, args.output_dir)

    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
