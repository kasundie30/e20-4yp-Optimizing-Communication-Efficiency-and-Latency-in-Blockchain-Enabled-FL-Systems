"""
xai/explainer.py — SHAP and LIME explanations for Aljunaid et al. (2025).

Paper §5.4 + Eqs. 3 & 4:
  - SHAP (Shapley Additive Explanations): feature importance via Eq. 3
      phi_j = sum_{S subset F\{j}} [|S|!(|F|-|S|-1)!/|F|!] * [f(S u {j}) - f(S)]
  - LIME (Local Interpretable Model-agnostic Explanations): per-sample via Eq. 4
      fˆ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)

GBM → SHAP TreeExplainer (exact, fast)
SVM/LR → SHAP KernelExplainer (model-agnostic, slower)
All models → LIME LimeTabularExplainer
"""
import warnings
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap(model, X_background: np.ndarray, X_explain: np.ndarray,
             feature_names: list, output_dir: str):
    """
    Compute SHAP values and save summary plot.

    Parameters
    ----------
    model          : fitted sklearn estimator
    X_background   : background dataset (subsample for KernelExplainer)
    X_explain      : samples to explain
    feature_names  : list of feature name strings
    output_dir     : directory where plots are saved
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[XAI] shap or matplotlib not installed — skipping SHAP.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    model_type = type(model).__name__

    print(f"[XAI] Running SHAP ({model_type}) on {len(X_explain)} samples ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if "GradientBoosting" in model_type:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_explain)
                # shap_values may be 1-D array for binary classifiers
                if isinstance(shap_values, list):
                    sv = shap_values[1]   # positive class
                else:
                    sv = shap_values
            else:
                # KernelExplainer: use background sample
                bg = X_background[:config.SHAP_BACKGROUND_SAMPLES]
                predict_fn = (model.predict_proba
                              if hasattr(model, "predict_proba")
                              else model.predict)
                explainer   = shap.KernelExplainer(predict_fn, bg)
                shap_values = explainer.shap_values(X_explain[:50], nsamples=100)
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values

            # Summary bar plot (Figure 16-equivalent)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_explain[:len(sv)],
                              feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance — {model_type}", fontsize=13)
            plt.tight_layout()
            bar_path = os.path.join(output_dir, f"shap_bar_{model_type}.png")
            plt.savefig(bar_path, dpi=120)
            plt.close()
            print(f"[XAI] SHAP bar plot → {bar_path}")

            # Beeswarm summary plot
            plt.figure(figsize=(10, 7))
            shap.summary_plot(sv, X_explain[:len(sv)],
                              feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary — {model_type}", fontsize=13)
            plt.tight_layout()
            bee_path = os.path.join(output_dir, f"shap_beeswarm_{model_type}.png")
            plt.savefig(bee_path, dpi=120)
            plt.close()
            print(f"[XAI] SHAP beeswarm → {bee_path}")

            # Mean |SHAP| per feature
            mean_abs = np.abs(sv).mean(axis=0)
            importance = dict(zip(feature_names, mean_abs.tolist()))
            top = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("[XAI] Top-5 SHAP features:")
            for fname, fval in top[:5]:
                print(f"       {fname:20s}: {fval:.4f}")
            return importance

        except Exception as e:
            print(f"[XAI] SHAP failed: {e}")
            return None


# ── LIME ──────────────────────────────────────────────────────────────────────

def run_lime(model, X_train: np.ndarray, X_explain: np.ndarray,
             y_explain: np.ndarray, feature_names: list, output_dir: str,
             n_samples: int = 3):
    """
    Run LIME on a few individual samples and save explanation plots.

    Parameters
    ----------
    model        : fitted sklearn estimator
    X_train      : training data (used to compute feature statistics)
    X_explain    : samples to explain (subset of test set)
    y_explain    : true labels for X_explain
    feature_names: feature name strings
    output_dir   : directory where plots are saved
    n_samples    : number of individual samples to explain
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[XAI] lime not installed — skipping LIME.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[XAI] Running LIME on {n_samples} samples ...")

    predict_fn = (model.predict_proba
                  if hasattr(model, "predict_proba")
                  else lambda x: np.column_stack([1 - model.predict(x),
                                                   model.predict(x)]))

    explainer = LimeTabularExplainer(
        training_data   = X_train,
        feature_names   = feature_names,
        class_names     = ["Not Fraud", "Fraud"],
        mode            = "classification",
        random_state    = config.RANDOM_SEED,
    )

    # Explain a fraud sample and a non-fraud sample
    fraud_idx    = np.where(y_explain == 1)[0]
    nonfraud_idx = np.where(y_explain == 0)[0]
    sample_indices = []
    if len(fraud_idx):
        sample_indices.append(int(fraud_idx[0]))
    if len(nonfraud_idx):
        sample_indices.append(int(nonfraud_idx[0]))
    # Fill up to n_samples with random
    rng = np.random.default_rng(config.RANDOM_SEED)
    extra = rng.integers(0, len(X_explain), size=max(0, n_samples - len(sample_indices)))
    sample_indices.extend(extra.tolist())
    sample_indices = sample_indices[:n_samples]

    for idx in sample_indices:
        instance = X_explain[idx]
        true_lbl = int(y_explain[idx])
        try:
            exp = explainer.explain_instance(
                data_row        = instance,
                predict_fn      = predict_fn,
                num_features    = config.LIME_NUM_FEATURES,
                num_samples     = config.LIME_NUM_SAMPLES,
            )
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME — Sample {idx} (True: {'Fraud' if true_lbl else 'Not Fraud'})",
                         fontsize=11)
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"lime_sample_{idx}.png")
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
            print(f"[XAI] LIME plot → {save_path}")
        except Exception as e:
            print(f"[XAI] LIME sample {idx} failed: {e}")
