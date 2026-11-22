"""
The1: Interactive Web Application for Automated Law Discovery.
Supports:
1. File Upload (CSV, Excel .xlsx, .xls, TSV).
2. Manual Data Entry (Interactive in-browser spreadsheet + Raw text paste).
3. 16 Canonical Physical Law Presets.
Automatic partitioning, OOD extrapolation verification, and LaTeX visualization.
"""

import os
import io
import tempfile
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from the1 import pipeline, PhysicsBenchmarkSuite

# Support for Hugging Face ZeroGPU runtime
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

# Initialize The1 pipeline
pipe = pipeline("the1")
laws_dict = PhysicsBenchmarkSuite.get_all_laws()
preset_names = list(laws_dict.keys())


# =============================================================================
# Helper & Parsing Functions
# =============================================================================
def parse_raw_text(text: str) -> pd.DataFrame:
    """Parses raw pasted text (comma, tab, or space separated) into a DataFrame."""
    if not text or not text.strip():
        raise ValueError("Pasted text is empty.")
    
    # Try reading with pandas auto-detection
    sio = io.StringIO(text.strip())
    try:
        df = pd.read_csv(sio, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(io.StringIO(text.strip()), sep=r"\s+", engine='python')
        
    df = df.dropna(how='all')
    return df


def execute_discovery_core(clean_df: pd.DataFrame,
                           feature_cols: list,
                           target_col: str,
                           split_strategy: str,
                           n_train_samples: int,
                           test_ratio: float,
                           noise_level: float,
                           lean_verify: bool = False):
    """Core discovery engine shared by file upload, manual spreadsheet, and text paste."""
    if len(clean_df) < 3:
        return "⚠️ Error: Dataset must contain at least 3 valid numeric rows.", "", "", None, None, None

    X_all = clean_df[feature_cols].to_numpy(dtype=float)
    y_all = clean_df[target_col].to_numpy(dtype=float)
    total_n = len(clean_df)

    # Splitting Strategies
    if split_strategy == "Few-Shot Mode (N ≤ 10 Points)":
        n_train = min(max(3, int(n_train_samples)), total_n - 1)
        idx = np.random.RandomState(42).permutation(total_n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]
    elif split_strategy == "Out-of-Distribution (OOD Extrapolation)":
        sorted_idx = np.argsort(y_all)
        n_val = max(1, int(total_n * (test_ratio / 100.0)))
        n_train = total_n - n_val
        train_idx = sorted_idx[:n_train]
        val_idx = sorted_idx[n_train:]
    else: # Random Split
        n_val = max(1, int(total_n * (test_ratio / 100.0)))
        n_train = total_n - n_val
        idx = np.random.RandomState(42).permutation(total_n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    if noise_level > 0.0:
        scale = np.std(y_train) if np.std(y_train) > 1e-6 else 1.0
        y_train = y_train + np.random.RandomState(42).normal(0, (noise_level/100.0) * scale, size=y_train.shape)

    # Discover Governing Law
    active_pipe = pipeline("the1", verify_with_lean=True) if lean_verify else pipe
    result = active_pipe.discover(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        var_names=feature_cols,
        target_name=target_col
    )

    discovered_latex = f"### Discovered Governing Equation:\n\n$${result.latex}$$"
    discovered_code = f"# Python Representation\n{target_col} = {result.law.str_repr}"

    val_r2 = result.metrics['val_r2']
    if val_r2 is not None and val_r2 >= 0.99:
        badge = "🟢 **Universal Invariant Law Discovered** (Exact Generalization)"
    elif val_r2 is not None and val_r2 >= 0.90:
        badge = "🟡 **High-Precision Law** (Strong Generalization)"
    else:
        badge = "🟠 **Empirical Fit** (Moderate Extrapolation Fidelity)"

    val_r2_str = f"`{val_r2:.6f}`" if val_r2 is not None else "N/A"
    val_rmse_str = f"`{result.metrics['val_rmse']:.4e}`" if result.metrics['val_rmse'] is not None else "N/A"

    verification = result.verification or {}
    if verification:
        status = verification.get('status', 'n/a')
        proven = verification.get('proven', 0)
        total = verification.get('total', 0)
        lean_section = (f"- **Lean 4 Hypothesis Checks:** `{status}` "
                        f"({proven}/{total} proven)")
        for hyp in verification.get('hypotheses', []):
            mark = "✅" if hyp.get('status') == 'proven' else "•"
            lean_section += (f"\n  - {mark} `{hyp.get('name')}` "
                             f"({hyp.get('kind')}): {hyp.get('status')}")
        if verification.get('hint'):
            lean_section += f"\n  - ℹ️ {verification['hint']}"
    else:
        lean_section = ""

    metrics_md = f"""
### 📊 Verification & Reliability Scorecard:
- **Status:** {badge}
- **Data Partitioning:** {len(train_idx)} Training pairs vs. {len(val_idx)} Unseen Validation pairs
- **Train R² Fit:** `{result.metrics['train_r2']:.6f}` (RMSE: `{result.metrics['train_rmse']:.4e}`)
- **Unseen Validation R² (OOD):** {val_r2_str} (OOD RMSE: {val_rmse_str})
- **Occam AST Complexity:** `{result.metrics['complexity']}` nodes
{lean_section}
"""

    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    if len(feature_cols) == 1:
        x_min = min(np.min(X_train[:, 0]), np.min(X_val[:, 0])) * 0.9
        x_max = max(np.max(X_train[:, 0]), np.max(X_val[:, 0])) * 1.1
        dense_x = np.linspace(max(x_min, 1e-4), x_max, 300).reshape(-1, 1)
        dense_pred = result.predict(dense_x)

        ax1.plot(dense_x, dense_pred, color='#e63946', linewidth=2.5, label=f'The1 Law: ${result.latex}$')
        ax1.scatter(X_train[:, 0], y_train, color='#1d3557', s=70, zorder=5, label=f'Train ({len(train_idx)} points)')
        ax1.scatter(X_val[:, 0], y_val, color='#2a9d8f', s=35, alpha=0.7, zorder=4, label=f'Unseen Val ({len(val_idx)} points)')
        ax1.set_xlabel(feature_cols[0], fontsize=11, fontweight='bold')
        ax1.set_ylabel(target_col, fontsize=11, fontweight='bold')
        ax1.set_title(f"Discovered Curve vs. Empirical Points", fontsize=12, fontweight='bold')
    else:
        pred_train = result.predict(X_train)
        pred_val = result.predict(X_val)

        ax1.scatter(y_train, pred_train, color='#1d3557', s=70, zorder=5, label=f'Train ({len(train_idx)} points)')
        ax1.scatter(y_val, pred_val, color='#2a9d8f', s=35, alpha=0.7, zorder=4, label=f'Unseen Val ({len(val_idx)} points)')

        all_y = np.concatenate([y_train, y_val])
        lims = [np.min(all_y), np.max(all_y)]
        ax1.plot(lims, lims, color='#e63946', linestyle='--', linewidth=2, label='Ideal Identity (y = y_hat)')
        ax1.set_xlabel(f"True {target_col}", fontsize=11, fontweight='bold')
        ax1.set_ylabel(f"Predicted {target_col}", fontsize=11, fontweight='bold')
        ax1.set_title("Parity Plot (Actual vs. Predicted)", fontsize=12, fontweight='bold')

    ax1.legend(frameon=True, fontsize=9)

    pred_all = result.predict(X_all)
    residuals = y_all - pred_all
    ax2.scatter(pred_all[train_idx], residuals[train_idx], color='#1d3557', s=50, label='Train Residuals')
    ax2.scatter(pred_all[val_idx], residuals[val_idx], color='#2a9d8f', s=35, alpha=0.7, label='Val Residuals')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel(f"Predicted {target_col}", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Residual Error (y - y_hat)", fontsize=11, fontweight='bold')
    ax2.set_title("Residual Error Analysis", fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()

    # Generate Export CSV
    export_df = clean_df.copy()
    export_df[f"{target_col}_predicted"] = np.round(pred_all, 6)
    export_df["residual_error"] = np.round(residuals, 6)
    export_df["split"] = ["Train" if i in train_idx else "Validation" for i in range(len(clean_df))]

    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    export_df.to_csv(temp_csv.name, index=False)

    return (
        discovered_latex,
        discovered_code,
        metrics_md,
        fig,
        temp_csv.name
    )


# =============================================================================
# Tab 1: File Upload Handler
# =============================================================================
def load_uploaded_sheet(file_obj):
    if file_obj is None:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=[]), "Please upload a valid CSV or Excel file."
    
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(file_path)

        df = df.dropna(how='all')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return df.head(10), gr.update(choices=[], value=None), gr.update(choices=[], value=[]), "⚠️ Error: Sheet must contain at least 2 numeric columns."

        default_target = numeric_cols[-1]
        default_features = [c for c in numeric_cols if c != default_target]
        info_msg = f"✅ Loaded **{len(df)} rows** and **{len(df.columns)} columns** ({len(numeric_cols)} numeric columns)."
        return df.head(10), gr.update(choices=numeric_cols, value=default_target), gr.update(choices=numeric_cols, value=default_features), info_msg
    except Exception as e:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=[]), f"⚠️ Error: {str(e)}"


@spaces.GPU
def discover_from_file_wrapper(file_obj, target_col, feature_cols, split_mode, n_train, test_ratio, noise, lean_verify):
    if file_obj is None:
        return "⚠️ Please upload a data sheet first.", "", "", None, None
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, sep=None, engine='python')
    except Exception as e:
        return f"⚠️ Error reading sheet: {e}", "", "", None, None

    if not target_col or not feature_cols:
        return "⚠️ Please select a Target column and at least one Feature column.", "", "", None, None

    clean_df = df[feature_cols + [target_col]].dropna()
    return execute_discovery_core(clean_df, feature_cols, target_col, split_mode, n_train, test_ratio, noise, lean_verify)


# =============================================================================
# Tab 2: Manual Data Entry & Interactive Table Handlers
# =============================================================================
SAMPLE_1D = pd.DataFrame({
    'length_L': [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    'period_T': [1.003, 1.418, 2.006, 2.457, 2.837, 3.172, 3.474, 4.012]
})

SAMPLE_2D = pd.DataFrame({
    'mass_m': [1.0, 2.0, 0.5, 3.0, 1.5, 4.0, 2.5, 0.8],
    'velocity_v': [3.0, 2.0, 4.0, 1.5, 5.0, 2.5, 3.0, 6.0],
    'energy_Ek': [4.5, 4.0, 4.0, 3.375, 18.75, 12.5, 11.25, 14.4]
})

SAMPLE_3D = pd.DataFrame({
    'm1': [1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 3.5, 5.0],
    'm2': [2.0, 3.0, 1.0, 4.0, 2.0, 1.5, 3.0, 2.0],
    'r':  [1.0, 2.0, 1.5, 2.5, 1.0, 3.0, 2.0, 2.0],
    'Force_F': [2.0, 1.5, 1.333, 0.96, 5.0, 0.667, 2.625, 2.5]
})


def load_manual_preset(sample_type: str):
    if sample_type == "1D (Pendulum T ~ sqrt(L))":
        df = SAMPLE_1D.copy()
    elif sample_type == "2D (Kinetic Energy E = 0.5*m*v^2)":
        df = SAMPLE_2D.copy()
    elif sample_type == "3D (Gravitation F ~ m1*m2/r^2)":
        df = SAMPLE_3D.copy()
    else:
        df = pd.DataFrame({'x1': [1.0, 2.0, 3.0], 'y': [2.0, 4.0, 6.0]})
        
    cols = df.columns.tolist()
    return df, gr.update(choices=cols, value=cols[-1]), gr.update(choices=cols, value=cols[:-1])


def import_pasted_text_to_table(text: str):
    try:
        df = parse_raw_text(text)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return df, gr.update(choices=[], value=None), gr.update(choices=[], value=[]), "⚠️ Error: Pasted data must contain at least 2 numeric columns."
        default_target = numeric_cols[-1]
        default_features = [c for c in numeric_cols if c != default_target]
        msg = f"✅ Pasted table imported: **{len(df)} rows** and **{len(numeric_cols)} numeric columns**."
        return df, gr.update(choices=numeric_cols, value=default_target), gr.update(choices=numeric_cols, value=default_features), msg
    except Exception as e:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=[]), f"⚠️ Error parsing text: {str(e)}"


@spaces.GPU
def discover_from_manual_table(manual_df, target_col, feature_cols, split_mode, n_train, test_ratio, noise, lean_verify):
    if manual_df is None or len(manual_df) == 0:
        return "⚠️ Table is empty. Please enter data or click a preset button.", "", "", None, None

    # Convert to numeric
    clean_df = pd.DataFrame(manual_df).apply(pd.to_numeric, errors='coerce').dropna()
    if not target_col or not feature_cols:
        cols = clean_df.columns.tolist()
        if len(cols) >= 2:
            target_col = cols[-1]
            feature_cols = cols[:-1]
        else:
            return "⚠️ Please select a target column and at least one feature column.", "", "", None, None

    return execute_discovery_core(clean_df, feature_cols, target_col, split_mode, n_train, test_ratio, noise, lean_verify)


# =============================================================================
# Tab 3: Preset Benchmark Laws Handler
# =============================================================================
@spaces.GPU
def run_preset_demo(preset_name: str, n_train: int, noise_pct: float):
    meta = laws_dict[preset_name]
    X_tr, y_tr = meta.sample_data(n_samples=n_train, noise_std=noise_pct/100.0, seed=42)
    X_val, y_val = meta.sample_data(n_samples=50, ood=True, seed=100)

    result = pipe.discover(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        var_names=meta.var_names,
        target_name=meta.target_name
    )

    discovered_eq = f"$${result.latex}$$"
    ground_truth_eq = f"$${meta.target_name} = {meta.formula_str}$$"
    metrics_summary = (
        f"**Domain:** {meta.category}\n\n"
        f"**Train R²:** `{result.metrics['train_r2']:.6f}` | **Unseen Val R² (OOD):** `{result.metrics['val_r2']:.6f}`\n\n"
        f"**AST Complexity:** `{result.metrics['complexity']}` nodes"
    )

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    if X_tr.shape[1] == 1:
        dense_x = np.linspace(np.min(X_tr)*0.9, np.max(X_val)*1.1, 250).reshape(-1, 1)
        ax.plot(dense_x, result.predict(dense_x), color='#e63946', linewidth=2.5, label=f'Law: ${result.latex}$')
        ax.scatter(X_tr[:, 0], y_tr, color='#1d3557', s=70, label='Train Pairs')
        ax.scatter(X_val[:, 0], y_val, color='#2a9d8f', s=25, alpha=0.6, label='Unseen Val (OOD)')
        ax.set_xlabel(meta.var_names[0])
        ax.set_ylabel(meta.target_name)
    else:
        all_y = np.concatenate([y_tr, y_val])
        ax.scatter(y_tr, result.predict(X_tr), color='#1d3557', s=70, label='Train')
        ax.scatter(y_val, result.predict(X_val), color='#2a9d8f', s=25, alpha=0.6, label='Unseen Val')
        ax.plot([min(all_y), max(all_y)], [min(all_y), max(all_y)], 'r--', label='Identity')
        ax.set_xlabel(f"True {meta.target_name}")
        ax.set_ylabel(f"Predicted {meta.target_name}")

    ax.set_title(f"Discovery for {meta.name}", fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()

    return discovered_eq, ground_truth_eq, metrics_summary, fig


# =============================================================================
# Main Gradio Interface
# =============================================================================
with gr.Blocks(title="The1: Universal Inductive World Model") as demo:
    gr.Markdown(
        r"""
        # 🌌 The1: Universal Inductive World Model
        ### Discover Closed-Form Mathematical Laws from Any Data (≤ 10 Pairs)
        
        Whether you **upload a spreadsheet file (CSV / Excel)**, **enter data manually in an interactive table**,
        or **paste raw text numbers**, **The1** partitions the dataset, induces the governing mathematical law,
        and verifies out-of-distribution (OOD) generalization!
        """
    )

    with gr.Tabs():
        # =====================================================================
        # TAB 1: FILE UPLOAD (CSV / Excel)
        # =====================================================================
        with gr.TabItem("📁 Option 1: Upload Data Sheet (CSV / Excel)"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Data Sheet (.csv, .xlsx, .xls, .tsv)",
                        file_types=[".csv", ".xlsx", ".xls", ".tsv"]
                    )
                    file_status = gr.Markdown()
                    
                    target_dropdown_f = gr.Dropdown(label="Target Column (y to predict)", choices=[])
                    features_dropdown_f = gr.Dropdown(label="Feature Columns (X inputs)", choices=[], multiselect=True)

                    with gr.Accordion("⚙️ Splitting & Induction Settings", open=True):
                        split_mode_f = gr.Radio(
                            choices=[
                                "Few-Shot Mode (N ≤ 10 Points)",
                                "Out-of-Distribution (OOD Extrapolation)",
                                "Random Train/Validation Split"
                            ],
                            value="Few-Shot Mode (N ≤ 10 Points)",
                            label="Data Partitioning Strategy"
                        )
                        n_train_slider_f = gr.Slider(minimum=3, maximum=30, value=8, step=1, label="Few-Shot Training Samples (N)")
                        test_ratio_slider_f = gr.Slider(minimum=10, maximum=50, value=30, step=5, label="Validation Set Ratio (%)")
                        noise_slider_f = gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.5, label="Noise Injection (%)")

                    discover_btn_f = gr.Button("⚡ Discover Law from File", variant="primary", size="lg")
                    lean_check_f = gr.Checkbox(
                        label="🧮 Check hypotheses in Lean 4 (requires Mathlib)",
                        value=False
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Sheet Preview")
                    data_preview_f = gr.DataFrame(interactive=False)
                    out_latex_f = gr.Markdown()
                    out_code_f = gr.Code(label="Discovered Formula", language="python")
                    out_scorecard_f = gr.Markdown()
                    out_figures_f = gr.Plot(label="Diagnostic Plots")
                    out_download_f = gr.File(label="📥 Download Dataset with Model Predictions")

            file_input.change(
                fn=load_uploaded_sheet,
                inputs=[file_input],
                outputs=[data_preview_f, target_dropdown_f, features_dropdown_f, file_status]
            )

            discover_btn_f.click(
                fn=discover_from_file_wrapper,
                inputs=[
                    file_input,
                    target_dropdown_f,
                    features_dropdown_f,
                    split_mode_f,
                    n_train_slider_f,
                    test_ratio_slider_f,
                    noise_slider_f,
                    lean_check_f
                ],
                outputs=[out_latex_f, out_code_f, out_scorecard_f, out_figures_f, out_download_f]
            )

        # =====================================================================
        # TAB 2: MANUAL DATA ENTRY (Interactive Table + Text Paste)
        # =====================================================================
        with gr.TabItem("✍️ Option 2: Enter Data Manually (Spreadsheet / Paste)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Interactive Table or Quick Template")
                    sample_buttons = gr.Radio(
                        choices=[
                            "1D (Pendulum T ~ sqrt(L))",
                            "2D (Kinetic Energy E = 0.5*m*v^2)",
                            "3D (Gravitation F ~ m1*m2/r^2)"
                        ],
                        value="2D (Kinetic Energy E = 0.5*m*v^2)",
                        label="Load Sample Template"
                    )

                    with gr.Accordion("📋 Or Paste Raw Text (Comma, Tab, or Space Separated)", open=False):
                        paste_box = gr.Textbox(
                            label="Paste your numbers here (with or without headers)",
                            placeholder="x1, x2, y\n1.0, 3.0, 4.5\n2.0, 2.0, 4.0\n0.5, 4.0, 4.0\n3.0, 1.5, 3.375",
                            lines=5
                        )
                        import_paste_btn = gr.Button("📥 Import Pasted Text into Table")
                        paste_status = gr.Markdown()

                    target_dropdown_m = gr.Dropdown(label="Target Column (y)", choices=['mass_m', 'velocity_v', 'energy_Ek'], value='energy_Ek')
                    features_dropdown_m = gr.Dropdown(label="Feature Columns (X)", choices=['mass_m', 'velocity_v', 'energy_Ek'], value=['mass_m', 'velocity_v'], multiselect=True)

                    with gr.Accordion("⚙️ Splitting Settings", open=True):
                        split_mode_m = gr.Radio(
                            choices=[
                                "Few-Shot Mode (N ≤ 10 Points)",
                                "Out-of-Distribution (OOD Extrapolation)",
                                "Random Train/Validation Split"
                            ],
                            value="Few-Shot Mode (N ≤ 10 Points)",
                            label="Partitioning Strategy"
                        )
                        n_train_slider_m = gr.Slider(minimum=3, maximum=20, value=6, step=1, label="Few-Shot Training Samples (N)")
                        test_ratio_slider_m = gr.Slider(minimum=10, maximum=50, value=25, step=5, label="Validation Ratio (%)")
                        noise_slider_m = gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.5, label="Noise (%)")

                    discover_btn_m = gr.Button("⚡ Discover Law from Manual Data", variant="primary", size="lg")
                    lean_check_m = gr.Checkbox(
                        label="🧮 Check hypotheses in Lean 4 (requires Mathlib)",
                        value=False
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### ✍️ Editable Spreadsheet (Double-click cells to edit or add rows)")
                    manual_table = gr.DataFrame(
                        value=SAMPLE_2D,
                        interactive=True,
                        label="Data Table (Click any cell to edit)"
                    )
                    out_latex_m = gr.Markdown()
                    out_code_m = gr.Code(label="Discovered Formula", language="python")
                    out_scorecard_m = gr.Markdown()
                    out_figures_m = gr.Plot(label="Diagnostic Plots")
                    out_download_m = gr.File(label="📥 Download Results CSV")

            sample_buttons.change(
                fn=load_manual_preset,
                inputs=[sample_buttons],
                outputs=[manual_table, target_dropdown_m, features_dropdown_m]
            )

            import_paste_btn.click(
                fn=import_pasted_text_to_table,
                inputs=[paste_box],
                outputs=[manual_table, target_dropdown_m, features_dropdown_m, paste_status]
            )

            discover_btn_m.click(
                fn=discover_from_manual_table,
                inputs=[
                    manual_table,
                    target_dropdown_m,
                    features_dropdown_m,
                    split_mode_m,
                    n_train_slider_m,
                    test_ratio_slider_m,
                    noise_slider_m,
                    lean_check_m
                ],
                outputs=[out_latex_m, out_code_m, out_scorecard_m, out_figures_m, out_download_m]
            )

        # =====================================================================
        # TAB 3: PRESET CANONICAL LAWS OF NATURE
        # =====================================================================
        with gr.TabItem("🌌 Option 3: 16 Canonical Physics Laws"):
            with gr.Row():
                with gr.Column(scale=1):
                    preset_dropdown = gr.Dropdown(
                        choices=preset_names,
                        value="kinetic_energy",
                        label="Select Physics Benchmark Law"
                    )
                    preset_n = gr.Slider(minimum=4, maximum=12, value=8, step=1, label="Training Pairs (N ≤ 10)")
                    preset_noise = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, step=0.5, label="Noise (%)")
                    demo_btn = gr.Button("⚡ Run Discovery Demo", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        p_disc = gr.Markdown()
                        p_true = gr.Markdown()
                    p_metrics = gr.Markdown()
                    p_plot = gr.Plot()

            demo_btn.click(
                fn=run_preset_demo,
                inputs=[preset_dropdown, preset_n, preset_noise],
                outputs=[p_disc, p_true, p_metrics, p_plot]
            )

        # =====================================================================
        # TAB 4: ABOUT & CITATION
        # =====================================================================
        with gr.TabItem("📖 About & Citation"):
            gr.Markdown(
                """
                ### About The1 Inductive World Model
                **The1** is an inductive neuro-symbolic World Model engineered to discover closed-form mathematical laws from extreme few-shot empirical data ($N \\le 10$).
                
                - **Hugging Face Model:** [racore/The1](https://huggingface.co/racore/The1)
                - **Author:** Loaii abdalslam (loaiabdalslam@gmail.com)
                
                ```bibtex
                @article{the1worldmodel2026,
                  title={The1: Inductive Neuro-Symbolic World Models for Extreme Few-Shot Natural Law Discovery},
                  author={Loaii abdalslam},
                  email={loaiabdalslam@gmail.com},
                  year={2026}
                }
                ```
                """
            )

if __name__ == "__main__":
    demo.launch()
