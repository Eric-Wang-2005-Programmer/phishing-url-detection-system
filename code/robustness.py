# ============================================================
# URL Robustness / Perturbation Experiment
#
# Frozen models:
#   CNN
#   MLP
#   Late Fusion
#   Intermediate Fusion
#
# IMPORTANT:
#   The SAME PhiUSIIL TEST URLs used by 01/02 are used here.
#   No retraining is performed.
#
# Perturbations:
#   Original
#   AddSlash
#   AddDot
#   AddHyphen
#   AddQuery
#   ChangeCase
#
# Delta:
#   Delta metric = Clean metric - Perturbed metric
# ============================================================

import os
import pickle
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from urllib.parse import (
    urlsplit, urlunsplit, parse_qsl, urlencode
)
from ipaddress import ip_address

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

import tensorflow as tf
from tensorflow.keras.models import load_model, Model


# ============================================================
# 1. Paths
# ============================================================

DRIVE_ROOT = "/content/drive/MyDrive"
PROJECT_ROOT = os.path.join(
    DRIVE_ROOT, "phishing_project"
)

CNN_DIR = os.path.join(PROJECT_ROOT, "cnn")
MLP_DIR = os.path.join(PROJECT_ROOT, "uci_mlp")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "baseline")
FUSION_DIR = os.path.join(PROJECT_ROOT, "fusion")
ROBUST_DIR = os.path.join(PROJECT_ROOT, "robustness")

UCI_PATH = '/content/drive/MyDrive/phishing_project/uci_mlp/uci-ml-phishing-dataset.csv'
PHIUSIIL_PATH = os.path.join(PROJECT_ROOT, "PhiUSIIL_Phishing_URL_Dataset.csv")

os.makedirs(ROBUST_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 2. Metrics
# ============================================================

def evaluate(y_true, prob):
    pred = (prob >= 0.5).astype(int)

    out = {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, pred, zero_division=0
        )
    }

    try:
        out["AUC"] = roc_auc_score(
            y_true, prob
        )
    except ValueError:
        out["AUC"] = np.nan

    return out


# ============================================================
# 3. Perturbation functions
# ============================================================

def ensure_scheme(url):
    if "://" in url:
        return url
    return "http://" + url


def split_url(url):
    u = ensure_scheme(str(url))
    return urlsplit(u)


def add_slash(url):
    p = split_url(url)
    path = p.path or "/"

    if not path.endswith("/"):
        path = path + "/"
    else:
        path = path + "/"

    return urlunsplit(
        (p.scheme, p.netloc, path, p.query, p.fragment)
    )


def add_dot(url):
    """
    Add a dot to the path rather than changing the registered
    domain. This is a controlled lexical/path perturbation.
    """
    p = split_url(url)
    path = p.path or "/"

    if path.endswith("/"):
        path = path + "."
    else:
        path = path + "."

    return urlunsplit(
        (p.scheme, p.netloc, path, p.query, p.fragment)
    )


def add_hyphen(url):
    """
    Add '-' to a query value when a query exists; otherwise
    create a harmless-looking query parameter.
    This avoids changing the registered domain itself.
    """
    p = split_url(url)

    if p.query:
        pairs = parse_qsl(
            p.query,
            keep_blank_values=True
        )

        if pairs:
            k, v = pairs[0]
            pairs[0] = (k, v + "-")
            query = urlencode(pairs)
        else:
            query = "x=-"
    else:
        query = "x=-"

    return urlunsplit(
        (p.scheme, p.netloc, p.path, query, p.fragment)
    )


def add_query(url):
    p = split_url(url)

    pairs = parse_qsl(
        p.query,
        keep_blank_values=True
    )

    pairs.append(
        ("tracking_id", "123")
    )

    query = urlencode(pairs)

    return urlunsplit(
        (p.scheme, p.netloc, p.path, query, p.fragment)
    )


def change_case(url):
    """
    Change only scheme + hostname case.
    Hostnames are case-insensitive; path case is NOT changed.
    """
    p = split_url(url)

    scheme = p.scheme.upper()

    netloc = p.netloc

    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        prefix = userinfo + "@"
    else:
        hostport = netloc
        prefix = ""

    # Preserve port if present.
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)
        if port.isdigit():
            hostport = host.upper() + ":" + port
        else:
            hostport = hostport.upper()
    else:
        hostport = hostport.upper()

    return urlunsplit(
        (
            scheme,
            prefix + hostport,
            p.path,
            p.query,
            p.fragment
        )
    )


PERTURBATIONS = {
    "Original": lambda x: x,
    "AddSlash": add_slash,
    "AddDot": add_dot,
    "AddHyphenQuery": add_hyphen,
    "AddQuery": add_query,
    "ChangeCase": change_case
}


# ============================================================
# 4. Same URL -> UCI feature adapter
# ============================================================

SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "bit.do", "cutt.ly",
    "tiny.cc", "lnkd.in", "rb.gy", "shorturl.at",
    "rebrand.ly", "shorte.st", "cli.gs", "bc.vc"
}


def safe_parts(url):
    u = ensure_scheme(str(url))

    try:
        p = urlsplit(u)
        return (
            (p.scheme or "").lower(),
            (p.hostname or "").lower(),
            p.path or ""
        )
    except Exception:
        return "", "", ""


def is_ip_host(host):
    try:
        ip_address(host)
        return True
    except Exception:
        return False


def subdomain_count(host):
    parts = [x for x in host.split(".") if x]
    return max(0, len(parts) - 2)


def normalize_col_name(name):
    return (
        str(name).strip().lower()
        .replace(" ", "")
        .replace("-", "_")
    )


def url_feature_values(url):
    url = str(url)
    scheme, host, path = safe_parts(url)

    n = len(url)

    if n < 54:
        url_len = 1
    elif n <= 75:
        url_len = 0
    else:
        url_len = -1

    subs = subdomain_count(host)

    if subs == 0:
        sub = 1
    elif subs == 1:
        sub = 0
    else:
        sub = -1

    rest = url.split("://", 1)[1] if "://" in url else url

    return {
        "having_ip_address":
            -1 if is_ip_host(host) else 1,

        "url_length":
            url_len,

        "shortining_service":
            -1 if host in SHORTENING_SERVICES else 1,

        "having_at_symbol":
            -1 if "@" in url else 1,

        "double_slash_redirecting":
            -1 if "//" in rest else 1,

        "prefix_suffix":
            -1 if "-" in host else 1,

        "having_sub_domain":
            sub,

        "https_token":
            -1 if "https" in host else 1,

        "abnormal_url": 1,
        "redirect": 1,
        "submitting_to_email": 1,
        "on_mouseover": 1,
        "rightclick": 1,
        "popupwidnow": 1,
        "iframe": 1
    }


def build_uci_adapter(urls, columns, scaler, reference_df):
    means = reference_df[columns].mean(
        numeric_only=True
    )

    rows = []

    for url in urls:
        uv = url_feature_values(url)
        row = {}

        for col in columns:
            key = normalize_col_name(col)

            if key in uv:
                row[col] = uv[key]
            else:
                row[col] = means[col]

        rows.append(row)

    X = pd.DataFrame(
        rows,
        columns=columns
    )

    return X


# ============================================================
# 5. Load all frozen artifacts
# ============================================================

cnn = load_model(
    os.path.join(
        CNN_DIR,
        "malicious_url_cnn.keras"
    )
)

with open(
    os.path.join(
        CNN_DIR,
        "tokenizer.pkl"
    ),
    "rb"
) as f:
    tokenizer = pickle.load(f)

with open(
    os.path.join(
        CNN_DIR,
        "label_encoder.pkl"
    ),
    "rb"
) as f:
    label_encoder = pickle.load(f)

mlp = joblib.load(
    os.path.join(
        MLP_DIR,
        "uci_mlp_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MLP_DIR,
        "uci_scaler.pkl"
    )
)

# Experiment 1 late fusion is truly zero-shot:
# fixed 50/50 weighting; no PhiUSIIL validation tuning.
alpha = 0.50

# Experiment 2 adaptation heads.
cnn_adapt_model = load_model(
    os.path.join(
        FUSION_DIR,
        "cnn_adaptation_head.keras"
    )
)

mlp_adapt_model = load_model(
    os.path.join(
        FUSION_DIR,
        "mlp_adaptation_head.keras"
    )
)

fusion_model = load_model(
    os.path.join(
        FUSION_DIR,
        "intermediate_fusion_adaptation.keras"
    )
)


# ============================================================
# 6. Load PhiUSIIL exact TEST split
# ============================================================

phi = pd.read_csv(PHIUSIIL_PATH)

label_candidates = [
    c for c in phi.columns
    if normalize_col_name(c) == "label"
]

label_col = label_candidates[0]

urls_all = phi["URL"].astype(str).values

# 1 legitimate / 0 phishing
y_all = (
    phi[label_col].astype(int) == 0
).astype(int).values

test_idx = np.load(
    os.path.join(
        BASELINE_DIR,
        "phi_test_idx.npy"
    )
)

urls_clean = urls_all[test_idx]
y_test = y_all[test_idx]


# ============================================================
# 7. Prepare UCI reference means
# ============================================================

uci_df = pd.read_csv(UCI_PATH)

if "id" in uci_df.columns:
    uci_df = uci_df.drop(columns=["id"])

uci_columns = [
    c for c in uci_df.columns
    if c != "Result"
]

reference_df, _ = __import__(
    "sklearn.model_selection",
    fromlist=["train_test_split"]
).train_test_split(
    uci_df,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=uci_df["Result"]
)


# ============================================================
# 8. CNN helpers
# ============================================================

MAX_LENGTH = 150

def tokenize_urls(urls):
    seq = tokenizer.texts_to_sequences(
        pd.Series(urls).astype(str)
    )

    return tf.keras.utils.pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )


names = [
    str(x).lower()
    for x in label_encoder.classes_
]

phishing_indices = [
    i for i, name in enumerate(names)
    if "phish" in name
]

if not phishing_indices:
    raise ValueError(
        "Saved CNN does not contain a phishing class."
    )

phishing_index = phishing_indices[0]


pool_layer = None

for layer in cnn.layers:
    if layer.__class__.__name__ == "GlobalMaxPooling1D":
        pool_layer = layer
        break

if pool_layer is None:
    raise ValueError(
        "GlobalMaxPooling1D layer not found."
    )

from tensorflow.keras import Input
from tensorflow.keras.models import Model

MAX_LENGTH = 150

cnn_input = Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="cnn_feature_input"
)

x = cnn_input
pool_output = None

for layer in cnn.layers:

    if layer.__class__.__name__ == "InputLayer":
        continue

    x = layer(x)

    if layer.__class__.__name__ == "GlobalMaxPooling1D":
        pool_output = x

if pool_output is None:
    raise ValueError(
        "GlobalMaxPooling1D layer was not found in the saved CNN.\n"
        "CNN layers:\n" +
        "\n".join(
            f"{i}: {layer.name} ({layer.__class__.__name__})"
            for i, layer in enumerate(cnn.layers)
        )
    )

cnn_embedding_model = Model(
    inputs=cnn_input,
    outputs=pool_output,
    name="cnn_embedding_extractor"
)

print("CNN embedding extractor created successfully.")
print("Embedding dimension:", cnn_embedding_model.output_shape)


def cnn_predict(urls):
    X = tokenize_urls(urls)
    p = cnn.predict(
        X,
        batch_size=512,
        verbose=0
    )
    return p[:, phishing_index]


def cnn_embed(urls):
    X = tokenize_urls(urls)
    return cnn_embedding_model.predict(
        X,
        batch_size=512,
        verbose=0
    )


# ============================================================
# 9. MLP helpers
# ============================================================

def relu(x):
    return np.maximum(x, 0.0)


def mlp_hidden_rep(model, X):
    A = X

    for i in range(len(model.coefs_) - 1):
        A = (
            A @ model.coefs_[i]
            + model.intercepts_[i]
        )
        A = relu(A)

    return A


def mlp_predict_and_embed(urls):
    X = build_uci_adapter(
        urls,
        uci_columns,
        scaler,
        reference_df
    )

    X_scaled = scaler.transform(X)

    proba = mlp.predict_proba(
        X_scaled
    )

    # User's original ppu_dl.py uses:
    # y = (Result == 1).astype(int)
    # UCI original +1 = legitimate, -1 = phishing.
    p_phishing = proba[:, 0]

    hidden = mlp_hidden_rep(
        mlp,
        X_scaled
    )

    return p_phishing, hidden


# ============================================================
# 10. Prediction pipeline for one perturbation
# ============================================================

def predict_all_models(urls):
    # -------------------------
    # Experiment 1: zero-shot
    # -------------------------

    p_cnn_zero = cnn_predict(urls)

    p_mlp_zero, mlp_hidden = (
        mlp_predict_and_embed(urls)
    )

    # Fixed 50/50 late fusion.
    p_late = (
        alpha * p_cnn_zero
        + (1.0 - alpha) * p_mlp_zero
    )

    # -------------------------
    # Frozen representations
    # -------------------------

    c_emb = cnn_embed(urls)

    H = np.concatenate(
        [c_emb, mlp_hidden],
        axis=1
    )

    # -------------------------
    # Experiment 2:
    # small adaptation heads
    # -------------------------

    p_cnn_adapt = (
        cnn_adapt_model.predict(
            c_emb,
            batch_size=512,
            verbose=0
        ).reshape(-1)
    )

    p_mlp_adapt = (
        mlp_adapt_model.predict(
            mlp_hidden,
            batch_size=512,
            verbose=0
        ).reshape(-1)
    )

    p_intermediate = (
        fusion_model.predict(
            H,
            batch_size=512,
            verbose=0
        ).reshape(-1)
    )

    return {
        "CNN_ZeroShot":
            p_cnn_zero,
        "MLP_ZeroShot":
            p_mlp_zero,
        "Late_Fusion_ZeroShot":
            p_late,
        "CNN_Adaptation":
            p_cnn_adapt,
        "MLP_Adaptation":
            p_mlp_adapt,
        "Intermediate_Fusion":
            p_intermediate
    }


# ============================================================
# 11. Run clean + perturbations
# ============================================================

all_results = []
all_predictions = {}

print("=" * 70)
print("03 ROBUSTNESS EXPERIMENT")
print("=" * 70)
print("Test samples:", len(urls_clean))

for perturb_name, perturb_fn in PERTURBATIONS.items():

    print("\nRunning:", perturb_name)

    perturbed_urls = np.array([
        perturb_fn(u)
        for u in urls_clean
    ])

    predictions = predict_all_models(
        perturbed_urls
    )

    all_predictions[
        perturb_name
    ] = predictions

    for model_name, prob in predictions.items():

        metrics = evaluate(
            y_test,
            prob
        )

        all_results.append({
            "Perturbation": perturb_name,
            "Model": model_name,
            **metrics
        })

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# 12. Save robustness results
# ============================================================

results_path = os.path.join(
    ROBUST_DIR,
    "robustness_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 13. Compute degradation
# ============================================================

clean_df = results_df[
    results_df["Perturbation"] == "Original"
].copy()

clean_df = clean_df.set_index(
    "Model"
)

degradation_rows = []

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]

for perturbation in PERTURBATIONS:

    if perturbation == "Original":
        continue

    pert_df = results_df[
        results_df["Perturbation"] == perturbation
    ].set_index("Model")

    for model_name in pert_df.index:

        row = {
            "Perturbation": perturbation,
            "Model": model_name
        }

        for metric in metrics:
            clean_value = clean_df.loc[model_name, metric]
            perturbed_value = pert_df.loc[model_name, metric]

            row[f"Delta_{metric}"] = (
                clean_value - perturbed_value
            )

        # Relative F1 degradation makes drops comparable
        # across models with different clean F1 values.
        clean_f1 = clean_df.loc[model_name, "F1"]
        perturbed_f1 = pert_df.loc[model_name, "F1"]

        if clean_f1 != 0:
            row["Relative_F1_Drop"] = (
                (clean_f1 - perturbed_f1) / clean_f1
            )
        else:
            row["Relative_F1_Drop"] = np.nan

        degradation_rows.append(row)

degradation_df = pd.DataFrame(
    degradation_rows
)

degradation_path = os.path.join(
    ROBUST_DIR,
    "robustness_degradation.csv"
)

degradation_df.to_csv(
    degradation_path,
    index=False
)


# ============================================================
# 14. Save perturbed URLs
# ============================================================

perturbed_table = pd.DataFrame({
    "row_index": test_idx,
    "original_URL": urls_clean,
    "true_label": y_test
})

for perturbation, fn in PERTURBATIONS.items():
    if perturbation == "Original":
        continue

    perturbed_table[
        perturbation
    ] = [
        fn(u)
        for u in urls_clean
    ]

perturbed_table.to_csv(
    os.path.join(
        ROBUST_DIR,
        "perturbed_test_urls.csv"
    ),
    index=False
)


# ============================================================
# 15. Plot F1 degradation
# ============================================================

plot_df = degradation_df.pivot(
    index="Perturbation",
    columns="Model",
    values="Delta_F1"
)

ax = plot_df.plot(
    kind="bar",
    figsize=(12, 6)
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_ylabel(
    "F1 degradation (clean - perturbed)"
)

ax.set_title(
    "Robustness: F1 Degradation"
)

plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        ROBUST_DIR,
        "f1_degradation.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 16. Plot perturbed F1
# ============================================================

f1_plot = results_df.pivot(
    index="Perturbation",
    columns="Model",
    values="F1"
)

ax = f1_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

ax.set_ylim(0, 1)
ax.set_ylabel("F1")
ax.set_title(
    "Robustness: F1 under URL Perturbations"
)

plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        ROBUST_DIR,
        "f1_under_perturbations.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 17. Save configuration
# ============================================================

robust_config = {
    "test_split": "Same frozen PhiUSIIL test split as 01_baseline.py",
    "models": [
        "CNN_ZeroShot",
        "MLP_ZeroShot",
        "Late_Fusion_ZeroShot",
        "CNN_Adaptation",
        "MLP_Adaptation",
        "Intermediate_Fusion"
    ],
    "perturbations": list(PERTURBATIONS.keys()),
    "perturbation_type": "controlled URL lexical perturbations",
    "add_hyphen_definition": (
        "Hyphen inserted into a query value without changing "
        "the registered domain"
    ),
    "late_fusion_alpha_cnn": alpha,
    "late_fusion_selection": "fixed_0.5_zero_shot",
    "retraining": False,
    "adaptation_backbones_frozen": True,
    "delta_definition":
        "clean_metric - perturbed_metric",
    "relative_f1_drop_definition":
        "(clean_F1 - perturbed_F1) / clean_F1",
    "label_definition": {
        "0": "legitimate",
        "1": "phishing"
    }
}

with open(
    os.path.join(
        ROBUST_DIR,
        "robustness_config.json"
    ),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        robust_config,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 70)
print("03 ROBUSTNESS COMPLETE")
print("=" * 70)
print("\nRaw results:")
print(results_path)
print("\nDegradation:")
print(degradation_path)
print("\nOutput directory:")
print(ROBUST_DIR)
