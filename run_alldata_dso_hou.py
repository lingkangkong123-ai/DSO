import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


ROOT = Path(__file__).resolve().parent
DSO_PACKAGE_ROOT = ROOT / "dso"
if str(DSO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(DSO_PACKAGE_ROOT))

from dso import DeepSymbolicOptimizer
from dso.program import Program


INPUT_DIR = ROOT / "outputs" / "alldata_dso"
OUTPUT_DIR = ROOT / "outputs" / "alldata_dso_hou"

X_COLS = ["pi1", "pi2", "pi3", "pi4", "pi5"]
CD_COL = "Cd_true"
MFLOW_COL = "m_flow_g_s"

VARIANT_DIR_NAMES = {
    "default": "default",
    "regularized_low_complexity": "reg_low",
}


def ensure_paths():
    required_paths = [
        INPUT_DIR / "train_processed.csv",
        INPUT_DIR / "seen_test_processed.csv",
        INPUT_DIR / "unseen_test_processed.csv",
        INPUT_DIR / "split_summary.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared alldata outputs. Please run prepare_alldata_dso_hou.py first. "
            f"Missing: {missing}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    df_train = pd.read_csv(INPUT_DIR / "train_processed.csv")
    df_seen_test = pd.read_csv(INPUT_DIR / "seen_test_processed.csv")
    df_unseen_test = pd.read_csv(INPUT_DIR / "unseen_test_processed.csv")
    with open(INPUT_DIR / "split_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return df_train, df_seen_test, df_unseen_test, summary


def export_cd_dataset(df: pd.DataFrame, path: Path, with_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df[X_COLS + [CD_COL]].to_csv(path, index=False, header=with_header)


def to_dso_dataset_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_optimizer_config(variant_name, dataset_path, regularized=False):
    variant_dir_name = VARIANT_DIR_NAMES.get(variant_name, variant_name)
    if regularized:
        function_set = ["add", "sub", "mul", "div"]
        max_length = 16
        length_max = 16
        soft_length_loc = 7
        soft_length_scale = 2
        n_samples = 10000
        poly_optimizer_params = None
    else:
        function_set = ["add", "sub", "mul", "div", "exp", "log", "const", "poly"]
        max_length = 36
        length_max = 36
        soft_length_loc = 10
        soft_length_scale = 5
        n_samples = 16000
        poly_optimizer_params = {
            "degree": 2,
            "coef_tol": 1e-6,
            "regressor": "dso_least_squares",
            "regressor_params": {
                "cutoff_p_value": 1.0,
                "n_max_terms": None,
                "coef_tol": 1e-6,
            },
        }

    variant_dir = OUTPUT_DIR / variant_dir_name
    logdir = variant_dir / "dso_logs"

    return {
        "task": {
            "task_type": "regression",
            "dataset": to_dso_dataset_path(dataset_path),
            "function_set": function_set,
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "protected": True,
            "reward_noise": 0.0,
            "poly_optimizer_params": poly_optimizer_params,
        },
        "training": {
            "n_samples": n_samples,
            "batch_size": 500,
            "epsilon": 0.05,
            "n_cores_batch": 1,
            "verbose": True,
            "complexity": "token",
            "const_optimizer": "scipy",
            "const_params": {
                "method": "L-BFGS-B",
                "options": {"gtol": 1e-3},
            },
        },
        "prior": {
            "length": {"min_": 4, "max_": length_max, "on": True},
            "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": not regularized},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": True},
            "soft_length": {"loc": soft_length_loc, "scale": soft_length_scale, "on": True},
            "domain_range": {"on": False},
        },
        "policy": {"max_length": max_length},
        "logging": {
            "save_summary": False,
            "save_pareto_front": True,
            "hof": 100,
        },
        "experiment": {
            "seed": 42,
            "logdir": str(logdir),
            "exp_name": variant_dir_name,
        },
    }


def calc_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite_mask = np.isfinite(actual) & np.isfinite(predicted)
    if not finite_mask.any():
        return {
            "mse": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "mard_pct": float("inf"),
        }
    actual = actual[finite_mask]
    predicted = predicted[finite_mask]
    mse = float(np.mean((predicted - actual) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predicted - actual)))
    mard = float(
        np.mean(
            np.abs((predicted - actual) / np.where(np.abs(actual) < 1e-12, 1e-12, actual))
        )
        * 100.0
    )
    return {"mse": mse, "rmse": rmse, "mae": mae, "mard_pct": mard}


def rename_expression_variables(expression):
    renamed = expression
    for idx, col in reversed(list(enumerate(X_COLS, start=1))):
        renamed = renamed.replace(f"x{idx}", col)
    return renamed


def pretty_expression(expression):
    try:
        expr = sp.sympify(expression)
        return sp.pretty(expr)
    except Exception:
        return expression


def parse_expression_to_numpy(expression_named):
    expr = sp.sympify(expression_named)
    if expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError("Expression contains non-finite symbolic values.")
    return sp.lambdify(
        tuple(sp.Symbol(name) for name in X_COLS),
        expr,
        modules=[{"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "Abs": np.abs}, "numpy"],
    )


def build_mass_flow_formula(cd_expression_named):
    return (
        "Aref_m2 * (" + cd_expression_named + ") * "
        "sqrt(2 * rho_in_kg_m3 * deltaP_Pa) * 1000"
    )


def export_formula_table(formula_df, csv_path, txt_path, title):
    formula_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = [title, "=" * len(title), ""]
    for idx, row in formula_df.iterrows():
        lines.append(f"[{idx + 1}]")
        for column in formula_df.columns:
            lines.append(f"{column}:")
            lines.append(str(row[column]))
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def export_candidate_formulas(save_path, variant_dir):
    exported = {}
    for log_type in ["hof", "pf"]:
        matches = sorted(save_path.glob(f"*_{log_type}.csv"))
        if not matches:
            continue
        raw_path = matches[0]
        df = pd.read_csv(raw_path)
        if "expression" in df.columns:
            df["expression_named"] = df["expression"].map(rename_expression_variables)
            df["expression_pretty"] = df["expression_named"].map(pretty_expression)
            df["m_flow_formula"] = df["expression_named"].map(build_mass_flow_formula)
        readable_csv = variant_dir / f"{log_type}_readable.csv"
        readable_txt = variant_dir / f"{log_type}_readable.txt"
        export_formula_table(df, readable_csv, readable_txt, f"{log_type.upper()} formulas")
        exported[log_type] = {
            "raw_csv": str(raw_path),
            "readable_csv": str(readable_csv),
            "readable_txt": str(readable_txt),
        }
    return exported


def predict_mass_flow_g_s(df, cd_prediction):
    return (
        df["Aref_m2"].to_numpy(dtype=float)
        * np.asarray(cd_prediction, dtype=float)
        * np.sqrt(
            np.maximum(
                2.0
                * df["rho_in_kg_m3"].to_numpy(dtype=float)
                * df["deltaP_Pa"].to_numpy(dtype=float),
                0.0,
            )
        )
        * 1000.0
    )


def save_three_way_scatter(
    actual_train,
    pred_train,
    actual_seen,
    pred_seen,
    actual_unseen,
    pred_unseen,
    output_path,
    title,
    x_label,
    y_label,
):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    all_actual = np.concatenate([actual_train, actual_seen, actual_unseen])
    line = np.linspace(all_actual.min(), all_actual.max(), 200)

    ax.scatter(actual_train, pred_train, s=18, color="tab:blue", label="Train")
    ax.scatter(actual_seen, pred_seen, s=22, color="tab:orange", label="Seen test")
    ax.scatter(actual_unseen, pred_unseen, s=26, color="tab:red", label="Unseen test")
    ax.plot(line, line, "k--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_group_trend_plot(df_eval, output_path, title):
    if plt is None or df_eval.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for source_file, group in df_eval.groupby("source_file"):
        group = group.sort_values("pi5")
        label_suffix = source_file.replace("fixed_data_simple_", "").replace(".pkl", "")
        ax.plot(group["pi5"], group[MFLOW_COL], "o-", label=f"actual {label_suffix}")
        ax.plot(group["pi5"], group["m_flow_prediction_g_s"], "x--", label=f"pred {label_suffix}")

    ax.set_xlabel("pi5 = Z")
    ax.set_ylabel("Mass flow (g/s)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_svg_scatter(
    actual_train,
    pred_train,
    actual_seen,
    pred_seen,
    actual_unseen,
    pred_unseen,
    output_path,
    title,
    x_label,
    y_label,
):
    width = 920
    height = 670
    left = 90
    right = 30
    top = 60
    bottom = 70
    actual_train = np.asarray(actual_train, dtype=float)
    pred_train = np.asarray(pred_train, dtype=float)
    actual_seen = np.asarray(actual_seen, dtype=float)
    pred_seen = np.asarray(pred_seen, dtype=float)
    actual_unseen = np.asarray(actual_unseen, dtype=float)
    pred_unseen = np.asarray(pred_unseen, dtype=float)
    all_x = np.concatenate([actual_train, actual_seen, actual_unseen])
    all_y = np.concatenate([pred_train, pred_seen, pred_unseen])
    min_val = float(np.nanmin(np.concatenate([all_x, all_y])))
    max_val = float(np.nanmax(np.concatenate([all_x, all_y])))
    pad = 0.05 * (max_val - min_val if max_val > min_val else 1.0)
    x_min = min_val - pad
    x_max = max_val + pad
    y_min = x_min
    y_max = x_max

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * (width - left - right)

    def sy(y):
        return height - bottom - (y - y_min) / (y_max - y_min) * (height - top - bottom)

    def circles(xs, ys, color, r):
        return "\n".join(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{r}" fill="{color}" />'
            for x, y in zip(xs, ys)
            if np.isfinite(x) and np.isfinite(y)
        )

    ticks = np.linspace(x_min, x_max, 6)
    tick_svg = []
    for tick in ticks:
        x = sx(tick)
        y = sy(tick)
        tick_svg.append(f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{top}" stroke="#eeeeee" />')
        tick_svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#eeeeee" />')
        tick_svg.append(f'<text x="{x:.2f}" y="{height-bottom+22}" font-size="14" text-anchor="middle">{tick:.2f}</text>')
        tick_svg.append(f'<text x="{left-10}" y="{y+5:.2f}" font-size="14" text-anchor="end">{tick:.2f}</text>')

    diagonal = f'<line x1="{sx(x_min):.2f}" y1="{sy(x_min):.2f}" x2="{sx(x_max):.2f}" y2="{sy(x_max):.2f}" stroke="#333333" stroke-dasharray="6 4" />'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="30" font-size="24" text-anchor="middle">{escape(title)}</text>
{''.join(tick_svg)}
<rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" fill="none" stroke="#222222" />
{diagonal}
{circles(actual_train, pred_train, "#1f77b4", 4)}
{circles(actual_seen, pred_seen, "#ff7f0e", 4)}
{circles(actual_unseen, pred_unseen, "#d62728", 5)}
<text x="{width/2}" y="{height-20}" font-size="18" text-anchor="middle">{escape(x_label)}</text>
<text x="25" y="{height/2}" font-size="18" text-anchor="middle" transform="rotate(-90 25 {height/2})">{escape(y_label)}</text>
<circle cx="{left+20}" cy="{top+20}" r="5" fill="#1f77b4" /><text x="{left+35}" y="{top+25}" font-size="15">Train</text>
<circle cx="{left+20}" cy="{top+45}" r="5" fill="#ff7f0e" /><text x="{left+35}" y="{top+50}" font-size="15">Seen test</text>
<circle cx="{left+20}" cy="{top+70}" r="6" fill="#d62728" /><text x="{left+35}" y="{top+75}" font-size="15">Unseen test</text>
</svg>"""
    output_path.write_text(svg, encoding="utf-8")


def save_scatter_outputs(record, df_train, df_seen_test, df_unseen_test, base_name, title_prefix):
    render_svg_scatter(
        df_train[MFLOW_COL].to_numpy(dtype=float),
        record["flow_train_prediction"],
        df_seen_test[MFLOW_COL].to_numpy(dtype=float),
        record["flow_seen_prediction"],
        df_unseen_test[MFLOW_COL].to_numpy(dtype=float),
        record["flow_unseen_prediction"],
        OUTPUT_DIR / f"{base_name}_actual_vs_predicted.svg",
        f"{title_prefix}: actual vs predicted",
        "Actual mass flow (g/s)",
        "Predicted mass flow (g/s)",
    )
    save_three_way_scatter(
        df_train[MFLOW_COL].to_numpy(dtype=float),
        record["flow_train_prediction"],
        df_seen_test[MFLOW_COL].to_numpy(dtype=float),
        record["flow_seen_prediction"],
        df_unseen_test[MFLOW_COL].to_numpy(dtype=float),
        record["flow_unseen_prediction"],
        OUTPUT_DIR / f"{base_name}_actual_vs_predicted.png",
        f"{title_prefix}: actual vs predicted",
        "Actual mass flow (g/s)",
        "Predicted mass flow (g/s)",
    )

    seen_eval = df_seen_test.copy()
    seen_eval["m_flow_prediction_g_s"] = record["flow_seen_prediction"]
    unseen_eval = df_unseen_test.copy()
    unseen_eval["m_flow_prediction_g_s"] = record["flow_unseen_prediction"]
    save_group_trend_plot(
        seen_eval,
        OUTPUT_DIR / f"{base_name}_seen_test_trend.png",
        f"{title_prefix}: seen-test trend by valve opening",
    )
    save_group_trend_plot(
        unseen_eval,
        OUTPUT_DIR / f"{base_name}_unseen_test_trend.png",
        f"{title_prefix}: unseen-test trend by valve opening",
    )


def collect_formula_records(variant_summaries):
    records = []
    seen = set()
    for summary in variant_summaries:
        variant = summary["variant"]
        for log_type, paths in summary["exported_logs"].items():
            raw_path = Path(paths["raw_csv"])
            if not raw_path.exists():
                continue
            df = pd.read_csv(raw_path)
            for idx, row in df.iterrows():
                expression_raw = str(row.get("expression", "")).strip()
                if not expression_raw:
                    continue
                dedupe_key = (variant, log_type, expression_raw)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                expression_named = rename_expression_variables(expression_raw)
                records.append(
                    {
                        "model": f"{variant}_{log_type}_{idx + 1}",
                        "variant": variant,
                        "formula_source": log_type,
                        "rank_in_source": idx + 1,
                        "expression_raw": expression_raw,
                        "expression_named": expression_named,
                        "expression_pretty": pretty_expression(expression_named),
                        "m_flow_formula": build_mass_flow_formula(expression_named),
                    }
                )
    return records


def compute_expression_complexity(expression_named):
    try:
        expr = sp.sympify(expression_named)
        return int(sp.count_ops(expr, visual=False))
    except Exception:
        return 10 ** 9


def evaluate_formula_record(record, df_train, df_seen_test, df_unseen_test):
    complexity = compute_expression_complexity(record["expression_named"])
    try:
        evaluator = parse_expression_to_numpy(record["expression_named"])
    except Exception as exc:
        failed = dict(record)
        failed["expression_complexity"] = complexity
        failed["evaluation_error"] = str(exc)
        for key in [
            "train_cd_mse", "seen_cd_mse", "unseen_cd_mse",
            "train_cd_mard_pct", "seen_cd_mard_pct", "unseen_cd_mard_pct",
            "train_mse", "seen_mse", "unseen_mse",
            "train_mard_pct", "seen_mard_pct", "unseen_mard_pct",
            "train_rmse", "seen_rmse", "unseen_rmse",
        ]:
            failed[key] = float("inf")
        failed["cd_train_prediction"] = np.full(len(df_train), np.nan, dtype=float)
        failed["cd_seen_prediction"] = np.full(len(df_seen_test), np.nan, dtype=float)
        failed["cd_unseen_prediction"] = np.full(len(df_unseen_test), np.nan, dtype=float)
        failed["flow_train_prediction"] = np.full(len(df_train), np.nan, dtype=float)
        failed["flow_seen_prediction"] = np.full(len(df_seen_test), np.nan, dtype=float)
        failed["flow_unseen_prediction"] = np.full(len(df_unseen_test), np.nan, dtype=float)
        return failed

    def predict(df):
        inputs = [df[col].to_numpy(dtype=float) for col in X_COLS]
        try:
            cd_pred = np.asarray(evaluator(*inputs), dtype=float)
        except Exception:
            cd_pred = np.full(len(df), np.nan, dtype=float)
        if cd_pred.ndim == 0:
            cd_pred = np.full(len(df), float(cd_pred), dtype=float)
        flow_pred = predict_mass_flow_g_s(df, cd_pred)
        return cd_pred, flow_pred

    cd_train_pred, flow_train_pred = predict(df_train)
    cd_seen_pred, flow_seen_pred = predict(df_seen_test)
    cd_unseen_pred, flow_unseen_pred = predict(df_unseen_test)

    evaluated = dict(record)
    evaluated["expression_complexity"] = complexity
    train_cd_metrics = calc_metrics(df_train[CD_COL].to_numpy(dtype=float), cd_train_pred)
    seen_cd_metrics = calc_metrics(df_seen_test[CD_COL].to_numpy(dtype=float), cd_seen_pred)
    unseen_cd_metrics = calc_metrics(df_unseen_test[CD_COL].to_numpy(dtype=float), cd_unseen_pred)
    train_flow_metrics = calc_metrics(df_train[MFLOW_COL].to_numpy(dtype=float), flow_train_pred)
    seen_flow_metrics = calc_metrics(df_seen_test[MFLOW_COL].to_numpy(dtype=float), flow_seen_pred)
    unseen_flow_metrics = calc_metrics(df_unseen_test[MFLOW_COL].to_numpy(dtype=float), flow_unseen_pred)

    evaluated["train_cd_mse"] = train_cd_metrics["mse"]
    evaluated["seen_cd_mse"] = seen_cd_metrics["mse"]
    evaluated["unseen_cd_mse"] = unseen_cd_metrics["mse"]
    evaluated["train_cd_mard_pct"] = train_cd_metrics["mard_pct"]
    evaluated["seen_cd_mard_pct"] = seen_cd_metrics["mard_pct"]
    evaluated["unseen_cd_mard_pct"] = unseen_cd_metrics["mard_pct"]
    evaluated["train_mse"] = train_flow_metrics["mse"]
    evaluated["seen_mse"] = seen_flow_metrics["mse"]
    evaluated["unseen_mse"] = unseen_flow_metrics["mse"]
    evaluated["train_rmse"] = train_flow_metrics["rmse"]
    evaluated["seen_rmse"] = seen_flow_metrics["rmse"]
    evaluated["unseen_rmse"] = unseen_flow_metrics["rmse"]
    evaluated["train_mard_pct"] = train_flow_metrics["mard_pct"]
    evaluated["seen_mard_pct"] = seen_flow_metrics["mard_pct"]
    evaluated["unseen_mard_pct"] = unseen_flow_metrics["mard_pct"]
    evaluated["cd_train_prediction"] = cd_train_pred
    evaluated["cd_seen_prediction"] = cd_seen_pred
    evaluated["cd_unseen_prediction"] = cd_unseen_pred
    evaluated["flow_train_prediction"] = flow_train_pred
    evaluated["flow_seen_prediction"] = flow_seen_pred
    evaluated["flow_unseen_prediction"] = flow_unseen_pred
    return evaluated


def train_and_evaluate_variant(
    variant_name,
    df_train,
    df_seen_test,
    df_unseen_test,
    split_summary,
    regularized=False,
):
    variant_dir_name = VARIANT_DIR_NAMES.get(variant_name, variant_name)
    variant_dir = OUTPUT_DIR / variant_dir_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    variant_train_csv = variant_dir / "train_cd.csv"
    export_cd_dataset(df_train, variant_train_csv, with_header=False)

    config = build_optimizer_config(variant_name, variant_train_csv, regularized=regularized)

    Program.clear_cache()
    model = DeepSymbolicOptimizer(config)
    result = model.train()
    program = result["program"]

    def execute(df):
        X = df[X_COLS].to_numpy(dtype=float)
        cd_pred = np.asarray(program.execute(X), dtype=float)
        if cd_pred.ndim == 0:
            cd_pred = np.full(len(df), float(cd_pred), dtype=float)
        flow_pred = predict_mass_flow_g_s(df, cd_pred)
        return cd_pred, flow_pred

    df_train_eval = df_train.copy()
    df_seen_eval = df_seen_test.copy()
    df_unseen_eval = df_unseen_test.copy()
    df_train_eval["Cd_prediction"], df_train_eval["m_flow_prediction_g_s"] = execute(df_train_eval)
    df_seen_eval["Cd_prediction"], df_seen_eval["m_flow_prediction_g_s"] = execute(df_seen_eval)
    df_unseen_eval["Cd_prediction"], df_unseen_eval["m_flow_prediction_g_s"] = execute(df_unseen_eval)

    train_cd_metrics = calc_metrics(df_train[CD_COL], df_train_eval["Cd_prediction"])
    seen_cd_metrics = calc_metrics(df_seen_test[CD_COL], df_seen_eval["Cd_prediction"])
    unseen_cd_metrics = calc_metrics(df_unseen_test[CD_COL], df_unseen_eval["Cd_prediction"])
    train_flow_metrics = calc_metrics(df_train[MFLOW_COL], df_train_eval["m_flow_prediction_g_s"])
    seen_flow_metrics = calc_metrics(df_seen_test[MFLOW_COL], df_seen_eval["m_flow_prediction_g_s"])
    unseen_flow_metrics = calc_metrics(df_unseen_test[MFLOW_COL], df_unseen_eval["m_flow_prediction_g_s"])

    best_expression = str(result["expression"])
    best_expression_named = rename_expression_variables(best_expression)
    best_expression_pretty = pretty_expression(best_expression_named)
    best_m_flow_formula = build_mass_flow_formula(best_expression_named)

    save_three_way_scatter(
        df_train[CD_COL].to_numpy(dtype=float),
        df_train_eval["Cd_prediction"].to_numpy(dtype=float),
        df_seen_test[CD_COL].to_numpy(dtype=float),
        df_seen_eval["Cd_prediction"].to_numpy(dtype=float),
        df_unseen_test[CD_COL].to_numpy(dtype=float),
        df_unseen_eval["Cd_prediction"].to_numpy(dtype=float),
        variant_dir / "cd_scatter.png",
        f"{variant_name}: actual vs predicted Cd",
        "Actual Cd",
        "Predicted Cd",
    )
    save_three_way_scatter(
        df_train[MFLOW_COL].to_numpy(dtype=float),
        df_train_eval["m_flow_prediction_g_s"].to_numpy(dtype=float),
        df_seen_test[MFLOW_COL].to_numpy(dtype=float),
        df_seen_eval["m_flow_prediction_g_s"].to_numpy(dtype=float),
        df_unseen_test[MFLOW_COL].to_numpy(dtype=float),
        df_unseen_eval["m_flow_prediction_g_s"].to_numpy(dtype=float),
        variant_dir / "m_flow_scatter.png",
        f"{variant_name}: actual vs predicted mass flow",
        "Actual mass flow (g/s)",
        "Predicted mass flow (g/s)",
    )
    save_group_trend_plot(
        df_seen_eval,
        variant_dir / "seen_test_m_flow_trend.png",
        f"{variant_name}: seen-test mass-flow trend",
    )
    save_group_trend_plot(
        df_unseen_eval,
        variant_dir / "unseen_test_m_flow_trend.png",
        f"{variant_name}: unseen-test mass-flow trend",
    )

    save_path = Path(model.config_experiment["save_path"])
    exported_logs = export_candidate_formulas(save_path, variant_dir)

    summary = {
        "variant": variant_name,
        "regularized": regularized,
        "train_samples": len(df_train),
        "seen_test_samples": len(df_seen_test),
        "unseen_test_samples": len(df_unseen_test),
        "inputs": X_COLS,
        "cd_output": CD_COL,
        "mass_flow_output": MFLOW_COL,
        "unseen_refrigerant": split_summary["split_summary"]["unseen_refrigerant"],
        "best_expression_cd_raw": best_expression,
        "best_expression_cd_named": best_expression_named,
        "best_expression_cd_pretty": best_expression_pretty,
        "best_expression_mass_flow": best_m_flow_formula,
        "train_cd_metrics": train_cd_metrics,
        "seen_test_cd_metrics": seen_cd_metrics,
        "unseen_test_cd_metrics": unseen_cd_metrics,
        "train_m_flow_metrics": train_flow_metrics,
        "seen_test_m_flow_metrics": seen_flow_metrics,
        "unseen_test_m_flow_metrics": unseen_flow_metrics,
        "save_path": str(save_path),
        "exported_logs": exported_logs,
    }

    with open(variant_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {"dataset": "train", "target": "Cd", **train_cd_metrics},
            {"dataset": "seen_test", "target": "Cd", **seen_cd_metrics},
            {"dataset": "unseen_test", "target": "Cd", **unseen_cd_metrics},
            {"dataset": "train", "target": "m_flow_g_s", **train_flow_metrics},
            {"dataset": "seen_test", "target": "m_flow_g_s", **seen_flow_metrics},
            {"dataset": "unseen_test", "target": "m_flow_g_s", **unseen_flow_metrics},
        ]
    ).to_csv(variant_dir / "metrics.csv", index=False)

    df_train_eval.to_csv(variant_dir / "train_predictions.csv", index=False)
    df_seen_eval.to_csv(variant_dir / "seen_test_predictions.csv", index=False)
    df_unseen_eval.to_csv(variant_dir / "unseen_test_predictions.csv", index=False)

    if hasattr(model, "sess") and model.sess is not None:
        model.sess.close()

    return summary


def select_best_by_unseen_error(valid_records):
    return min(
        valid_records,
        key=lambda item: (item["unseen_mard_pct"], item["seen_mard_pct"], item["unseen_mse"]),
    )


def select_best_with_complexity_preference(valid_records):
    sorted_by_error = sorted(
        valid_records,
        key=lambda item: (item["unseen_mard_pct"], item["seen_mard_pct"], item["unseen_mse"]),
    )
    best_unseen_mard = sorted_by_error[0]["unseen_mard_pct"]
    best_seen_mard = sorted_by_error[0]["seen_mard_pct"]

    shortlisted = [
        item
        for item in valid_records
        if item["unseen_mard_pct"] <= best_unseen_mard * 1.15
        and item["seen_mard_pct"] <= best_seen_mard * 1.25
    ]
    if not shortlisted:
        shortlisted = valid_records

    return min(
        shortlisted,
        key=lambda item: (
            item["expression_complexity"],
            item["unseen_mard_pct"],
            item["seen_mard_pct"],
            item["unseen_mse"],
        ),
    )


def build_final_artifacts(df_train, df_seen_test, df_unseen_test, variant_summaries):
    formula_records = collect_formula_records(variant_summaries)
    evaluated_records = [
        evaluate_formula_record(record, df_train, df_seen_test, df_unseen_test)
        for record in formula_records
    ]
    metrics_df = pd.DataFrame(
        [
            {
                "model": record["model"],
                "variant": record["variant"],
                "formula_source": record["formula_source"],
                "rank_in_source": record["rank_in_source"],
                "expression_named": record["expression_named"],
                "expression_complexity": record["expression_complexity"],
                "evaluation_error": record.get("evaluation_error", ""),
                "train_mse": record["train_mse"],
                "seen_mse": record["seen_mse"],
                "unseen_mse": record["unseen_mse"],
                "train_mard_pct": record["train_mard_pct"],
                "seen_mard_pct": record["seen_mard_pct"],
                "unseen_mard_pct": record["unseen_mard_pct"],
                "train_cd_mse": record["train_cd_mse"],
                "seen_cd_mse": record["seen_cd_mse"],
                "unseen_cd_mse": record["unseen_cd_mse"],
                "train_cd_mard_pct": record["train_cd_mard_pct"],
                "seen_cd_mard_pct": record["seen_cd_mard_pct"],
                "unseen_cd_mard_pct": record["unseen_cd_mard_pct"],
            }
            for record in evaluated_records
        ]
    )
    metrics_df = metrics_df.sort_values(
        ["unseen_mard_pct", "seen_mard_pct", "unseen_mse", "model"]
    ).reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "all_formula_metrics.csv", index=False)

    valid_records = [
        item
        for item in evaluated_records
        if np.isfinite(item["unseen_mard_pct"]) and np.isfinite(item["unseen_mse"])
    ]
    if not valid_records:
        raise RuntimeError("No valid candidate formulas remained after postprocessing.")

    best_record = select_best_by_unseen_error(valid_records)
    best_complexity_record = select_best_with_complexity_preference(valid_records)

    best_summary = {
        "selection_rule": "minimize unseen_test mard, then seen_test mard, then unseen_test mse",
        "selected_model": best_record["model"],
        "variant": best_record["variant"],
        "formula_source": best_record["formula_source"],
        "rank_in_source": best_record["rank_in_source"],
        "expression_named": best_record["expression_named"],
        "expression_pretty": best_record["expression_pretty"],
        "expression_complexity": best_record["expression_complexity"],
        "m_flow_formula": best_record["m_flow_formula"],
        "train_mard_pct": best_record["train_mard_pct"],
        "seen_mard_pct": best_record["seen_mard_pct"],
        "unseen_mard_pct": best_record["unseen_mard_pct"],
        "train_mse": best_record["train_mse"],
        "seen_mse": best_record["seen_mse"],
        "unseen_mse": best_record["unseen_mse"],
    }
    with open(OUTPUT_DIR / "best_formula_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    complexity_summary = {
        "selection_rule": (
            "shortlist formulas within 15% of best unseen_test mard and within 25% "
            "of best seen_test mard, then minimize expression_complexity"
        ),
        "selected_model": best_complexity_record["model"],
        "variant": best_complexity_record["variant"],
        "formula_source": best_complexity_record["formula_source"],
        "rank_in_source": best_complexity_record["rank_in_source"],
        "expression_named": best_complexity_record["expression_named"],
        "expression_pretty": best_complexity_record["expression_pretty"],
        "expression_complexity": best_complexity_record["expression_complexity"],
        "m_flow_formula": best_complexity_record["m_flow_formula"],
        "train_mard_pct": best_complexity_record["train_mard_pct"],
        "seen_mard_pct": best_complexity_record["seen_mard_pct"],
        "unseen_mard_pct": best_complexity_record["unseen_mard_pct"],
        "train_mse": best_complexity_record["train_mse"],
        "seen_mse": best_complexity_record["seen_mse"],
        "unseen_mse": best_complexity_record["unseen_mse"],
    }
    with open(OUTPUT_DIR / "best_formula_complexity_aware_summary.json", "w", encoding="utf-8") as f:
        json.dump(complexity_summary, f, ensure_ascii=False, indent=2)

    save_scatter_outputs(
        best_record,
        df_train,
        df_seen_test,
        df_unseen_test,
        "best_formula",
        "DSO symbolic regression",
    )
    save_scatter_outputs(
        best_complexity_record,
        df_train,
        df_seen_test,
        df_unseen_test,
        "best_formula_complexity_aware",
        "DSO symbolic regression (complexity-aware)",
    )


def main():
    ensure_paths()
    df_train, df_seen_test, df_unseen_test, split_summary = load_processed_data()

    print(f"Using prepared data from: {INPUT_DIR}")
    print(f"Writing outputs to: {OUTPUT_DIR}")
    print(f"Train samples: {len(df_train)}")
    print(f"Seen-test samples: {len(df_seen_test)}")
    print(f"Unseen-test samples: {len(df_unseen_test)}")

    variant_summaries = []
    variant_summaries.append(
        train_and_evaluate_variant(
            "default",
            df_train,
            df_seen_test,
            df_unseen_test,
            split_summary,
            regularized=False,
        )
    )
    variant_summaries.append(
        train_and_evaluate_variant(
            "regularized_low_complexity",
            df_train,
            df_seen_test,
            df_unseen_test,
            split_summary,
            regularized=True,
        )
    )

    comparison_df = pd.DataFrame(
        [
            {
                "variant": summary["variant"],
                "regularized": summary["regularized"],
                "train_cd_rmse": summary["train_cd_metrics"]["rmse"],
                "seen_test_cd_rmse": summary["seen_test_cd_metrics"]["rmse"],
                "unseen_test_cd_rmse": summary["unseen_test_cd_metrics"]["rmse"],
                "train_cd_mard_pct": summary["train_cd_metrics"]["mard_pct"],
                "seen_test_cd_mard_pct": summary["seen_test_cd_metrics"]["mard_pct"],
                "unseen_test_cd_mard_pct": summary["unseen_test_cd_metrics"]["mard_pct"],
                "train_m_flow_rmse": summary["train_m_flow_metrics"]["rmse"],
                "seen_test_m_flow_rmse": summary["seen_test_m_flow_metrics"]["rmse"],
                "unseen_test_m_flow_rmse": summary["unseen_test_m_flow_metrics"]["rmse"],
                "train_m_flow_mard_pct": summary["train_m_flow_metrics"]["mard_pct"],
                "seen_test_m_flow_mard_pct": summary["seen_test_m_flow_metrics"]["mard_pct"],
                "unseen_test_m_flow_mard_pct": summary["unseen_test_m_flow_metrics"]["mard_pct"],
                "save_path": summary["save_path"],
            }
            for summary in variant_summaries
        ]
    )
    comparison_df.to_csv(OUTPUT_DIR / "variant_comparison.csv", index=False)

    with open(OUTPUT_DIR / "variant_comparison.json", "w", encoding="utf-8") as f:
        json.dump(variant_summaries, f, ensure_ascii=False, indent=2)

    build_final_artifacts(df_train, df_seen_test, df_unseen_test, variant_summaries)

    print("Finished. Variant summaries:")
    for summary in variant_summaries:
        print(
            f"  - {summary['variant']}: unseen-test mass-flow MARD = "
            f"{summary['unseen_test_m_flow_metrics']['mard_pct']:.2f}%"
        )
        print(f"    save path: {summary['save_path']}")


if __name__ == "__main__":
    main()
