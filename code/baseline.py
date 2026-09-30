# ============================================================

# Cross-dataset Baseline Evaluation
#
# IMPORTANT:
#   This script DOES NOT retrain the user's existing CNN / MLP.
#   It loads the artifacts produced by:
#       ml_dl.py
#       ppu_dl.py
#
# Experiment:
#   1. Load pretrained CNN + MLP
#   2. Load PhiUSIIL
#   3. Split PhiUSIIL: 75% / 15% / 10%
#   4. Evaluate CNN and MLP on the SAME PhiUSIIL test split
#   5. Save the exact split for 02_fusion.py / 03_robustness.py
#
# Label convention inside this project:
#   1 = phishing / malicious
#   0 = legitimate / benign
# ============================================================

import os
import json
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from urllib.parse import urlsplit
from ipaddress import ip_address

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# 1. Paths
# ============================================================

DRIVE_ROOT = "/content/drive/MyDrive"

MALICIOUS_PATH = '/content/drive/MyDrive/phishing_project/cnn/malicious_phish.csv'
UCI_PATH = '/content/drive/MyDrive/phishing_project/uci_mlp/uci-ml-phishing-dataset.csv'
PHIUSIIL_PATH = "/content/drive/MyDrive/phishing_project/PhiUSIIL_Phishing_URL_Dataset.csv"

PROJECT_ROOT = os.path.join(DRIVE_ROOT, "phishing_project")
CNN_DIR = os.path.join(PROJECT_ROOT, "cnn")
MLP_DIR = os.path.join(PROJECT_ROOT, "uci_mlp")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "baseline")

os.makedirs(BASELINE_DIR, exist_ok=True)

CNN_MODEL_PATH = os.path.join(CNN_DIR, "malicious_url_cnn.keras")
CNN_TOKENIZER_PATH = os.path.join(CNN_DIR, "tokenizer.pkl")
CNN_LABEL_ENCODER_PATH = os.path.join(CNN_DIR, "label_encoder.pkl")

MLP_MODEL_PATH = os.path.join(MLP_DIR, "uci_mlp_model.pkl")
MLP_SCALER_PATH = os.path.join(MLP_DIR, "uci_scaler.pkl")


# ============================================================
# 2. Reproducibility
# ============================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. Metrics
# ============================================================

def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

    try:
        result["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        result["AUC"] = np.nan

    return result, y_pred


# ============================================================
# 4. URL -> UCI feature adapter
# ============================================================

SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "bit.do", "cutt.ly",
    "tiny.cc", "lnkd.in", "rb.gy", "shorturl.at",
    "rebrand.ly", "shorte.st", "cli.gs", "bc.vc" 
}


def safe_parts(url):
    url = str(url).strip()

    if not url:
        return "", "", ""

    candidate = url
    if "://" not in candidate:
        candidate = "http://" + candidate

    try:
        parts = urlsplit(candidate)
        scheme = (parts.scheme or "").lower()
        hostname = (parts.hostname or "").lower()
        path = parts.path or ""
        return scheme, hostname, path
    except Exception:
        return "", "", ""


def is_ip_host(hostname):
    if not hostname:
        return False
    try:
        ip_address(hostname)
        return True
    except Exception:
        return False


def count_subdomain_levels(hostname):
    if not hostname:
        return 0

    host = hostname.split(":")[0]
    parts = [x for x in host.split(".") if x]

    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def uci_url_feature_values(url):
    """
    Return URL-observable UCI-style values {-1, 0, 1}.
    Only features directly observable from URL string are parsed here.
    Webpage/DOM features are excluded to allow build_uci_adapter 
    to fill them with UCI training-set means (Neutral Adapter).
    """
    url = str(url)
    scheme, host, path = safe_parts(url)
    lower_url = url.lower()

    values = {}

    # 1. having_IP_Address
    values["having_ip_address"] = -1 if is_ip_host(host) else 1

    # 2. url_length
    n = len(url)
    if n < 54:
        values["url_length"] = 1
    elif n <= 75:
        values["url_length"] = 0
    else:
        values["url_length"] = -1

    # 3. shortining_service
    values["shortining_service"] = -1 if host in SHORTENING_SERVICES else 1

    # 4. having_At_Symbol
    values["having_at_symbol"] = -1 if "@" in url else 1

    # 5. double_slash_redirecting
    rest = url.split("://", 1)[1] if "://" in url else url
    values["double_slash_redirecting"] = -1 if "//" in rest else 1

    # 6. Prefix_Suffix
    values["prefix_suffix"] = -1 if "-" in host else 1

    # 7. having_Sub_Domain
    sub_count = count_subdomain_levels(host)
    if sub_count == 0:
        values["having_sub_domain"] = 1
    elif sub_count == 1:
        values["having_sub_domain"] = 0
    else:
        values["having_sub_domain"] = -1

    # 8. HTTPS_token
    values["https_token"] = -1 if "https" in host else 1

    # 9. abnormal_url
    values["abnormal_url"] = -1 if host and host not in lower_url else 1

    return values


def normalize_col_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
    )


def build_uci_adapter(urls, uci_columns, scaler, uci_train_df):
    means = uci_train_df[uci_columns].mean(numeric_only=True)
    rows = []

    for url in urls:
        url_values = uci_url_feature_values(url)
        row = {}

        for col in uci_columns:
            key = normalize_col_name(col)

            if key in url_values:
                row[col] = url_values[key]
            else:
                row[col] = means[col]

        rows.append(row)

    X = pd.DataFrame(rows, columns=uci_columns)

    if X.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"MLP expects {scaler.n_features_in_} features, "
            f"but adapter produced {X.shape[1]}."
        )

    return X


# ============================================================
# 5. Load pretrained models
# ============================================================

print("=" * 70)
print("BASELINE: LOAD PRETRAINED CNN + MLP")
print("=" * 70)

required = [
    CNN_MODEL_PATH,
    CNN_TOKENIZER_PATH,
    CNN_LABEL_ENCODER_PATH,
    MLP_MODEL_PATH,
    MLP_SCALER_PATH,
    PHIUSIIL_PATH,
    UCI_PATH
]

for path in required:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Please run the user's original ml_dl.py / ppu_dl.py first."
        )

cnn = load_model(CNN_MODEL_PATH)

with open(CNN_TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(CNN_LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

mlp = joblib.load(MLP_MODEL_PATH)
scaler = joblib.load(MLP_SCALER_PATH)

print("CNN loaded:", CNN_MODEL_PATH)
print("MLP loaded:", MLP_MODEL_PATH)
print("CNN classes:", label_encoder.classes_)
print("MLP expected features:", scaler.n_features_in_)


# ============================================================
# 6. Load UCI feature schema
# ============================================================

uci_df = pd.read_csv(UCI_PATH)

if "Result" not in uci_df.columns:
    raise ValueError("UCI CSV does not contain Result column.")

if "id" in uci_df.columns:
    uci_df = uci_df.drop(columns=["id"])

uci_feature_columns = [c for c in uci_df.columns if c != "Result"]

if len(uci_feature_columns) != scaler.n_features_in_:
    raise ValueError(
        "The UCI CSV feature count does not match the saved scaler/model.\n"
        f"CSV features={len(uci_feature_columns)}, "
        f"model features={scaler.n_features_in_}"
    )

uci_train_reference, _ = train_test_split(
    uci_df,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=uci_df["Result"]
)


# ============================================================
# 7. Load PhiUSIIL
# ============================================================

print("\nLoading PhiUSIIL...")
phi = pd.read_csv(PHIUSIIL_PATH)

if "URL" not in phi.columns:
    raise ValueError("PhiUSIIL CSV must contain a URL column.")

label_candidates = [
    c for c in phi.columns
    if normalize_col_name(c) == "label"
]

if not label_candidates:
    raise ValueError("Could not find PhiUSIIL label column.")

label_col = label_candidates[0]

# Official PhiUSIIL convention: 1 = legitimate, 0 = phishing
# Internal project convention: 1 = phishing, 0 = legitimate
y_phi = (phi[label_col].astype(int) == 0).astype(int).values
urls_phi = phi["URL"].astype(str).values

print("PhiUSIIL shape:", phi.shape)
print("Label column:", label_col)
print("Internal labels: 1=phishing, 0=legitimate")
print(pd.Series(y_phi).value_counts().sort_index())


# ============================================================
# 8. Fixed 75/15/10 split
# ============================================================

all_idx = np.arange(len(phi))

train_idx, temp_idx = train_test_split(
    all_idx,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_phi
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=y_phi[temp_idx]
)

np.save(os.path.join(BASELINE_DIR, "phi_train_idx.npy"), train_idx)
np.save(os.path.join(BASELINE_DIR, "phi_val_idx.npy"), val_idx)
np.save(os.path.join(BASELINE_DIR, "phi_test_idx.npy"), test_idx)

print("\nPhiUSIIL split:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))


# ============================================================
# 9. CNN inference (Multi-class to Binary Malicious Probability)
# ============================================================

MAX_LENGTH = 150

def cnn_predict_phishing(urls):
    seq = tokenizer.texts_to_sequences(pd.Series(urls).astype(str))
    X = tf.keras.utils.pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probs = cnn.predict(X, batch_size=512, verbose=0)

    if probs.ndim != 2:
        raise ValueError("Unexpected CNN output shape.")

    class_names = [str(x).lower() for x in label_encoder.classes_]

    # Find benign (legitimate) class index
    benign_candidates = [
        i for i, name in enumerate(class_names)
        if "benign" in name
    ]

    if benign_candidates:
        # Malicious Probability = 1.0 - P(benign)
        # This covers ALL malicious types (phishing, malware, defacement)
        benign_index = benign_candidates[0]
        return 1.0 - probs[:, benign_index]
    else:
        # Fallback if benign label isn't present
        phishing_candidates = [
            i for i, name in enumerate(class_names)
            if "phish" in name
        ]
        if not phishing_candidates:
            raise ValueError(
                "CNN label encoder contains neither 'benign' nor 'phishing' class. "
                f"Classes are: {label_encoder.classes_}"
            )
        return probs[:, phishing_candidates[0]]


p_cnn = cnn_predict_phishing(urls_phi)

print("\nCNN probability range:")
print(float(p_cnn.min()), float(p_cnn.max()))


# ============================================================
# 10. MLP inference through UCI adapter
# ============================================================

print("\nBuilding UCI-feature adapter for PhiUSIIL...")

X_phi_uci = build_uci_adapter(
    urls_phi,
    uci_feature_columns,
    scaler,
    uci_train_reference
)

X_phi_uci_scaled = scaler.transform(X_phi_uci)
mlp_proba = mlp.predict_proba(X_phi_uci_scaled)

# Dynamic class index resolution for MLP (Class 0 represents Phishing)
if 0 in mlp.classes_:
    phishing_class_idx = np.where(mlp.classes_ == 0)[0][0]
else:
    phishing_class_idx = 0

p_mlp = mlp_proba[:, phishing_class_idx]

print("MLP probability range:")
print(float(p_mlp.min()), float(p_mlp.max()))


# ============================================================
# 11. Evaluate ONLY the held-out PhiUSIIL test set
# ============================================================

LATE_ALPHA = 0.50
p_late = LATE_ALPHA * p_cnn + (1.0 - LATE_ALPHA) * p_mlp

results = []

for name, prob in [
    ("CNN_ZeroShot", p_cnn),
    ("MLP_ZeroShot", p_mlp),
    ("Late_Fusion_ZeroShot", p_late)
]:
    metrics, pred = binary_metrics(y_phi[test_idx], prob[test_idx])

    row = {"Model": name}
    row.update(metrics)
    results.append(row)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(pd.Series(metrics))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_phi[test_idx], pred))


# ============================================================
# 12. Save baseline results
# ============================================================

results_df = pd.DataFrame(results)
results_path = os.path.join(BASELINE_DIR, "baseline_results.csv")
results_df.to_csv(results_path, index=False)

np.save(os.path.join(BASELINE_DIR, "phi_cnn_probability.npy"), p_cnn)
np.save(os.path.join(BASELINE_DIR, "phi_mlp_probability.npy"), p_mlp)
np.save(os.path.join(BASELINE_DIR, "phi_late_probability.npy"), p_late)
np.save(os.path.join(BASELINE_DIR, "phi_labels.npy"), y_phi)


# ============================================================
# 13. Save test prediction table
# ============================================================

test_table = pd.DataFrame({
    "row_index": test_idx,
    "URL": urls_phi[test_idx],
    "true_label": y_phi[test_idx],
    "cnn_probability_phishing": p_cnn[test_idx],
    "mlp_probability_phishing": p_mlp[test_idx]
})

test_table["cnn_prediction"] = (
    test_table["cnn_probability_phishing"] >= 0.5
).astype(int)

test_table["mlp_prediction"] = (
    test_table["mlp_probability_phishing"] >= 0.5
).astype(int)

test_table.to_csv(
    os.path.join(BASELINE_DIR, "phi_test_predictions.csv"),
    index=False
)


# ============================================================
# 14. Plot comparison
# ============================================================

plot_df = results_df.set_index("Model")

ax = plot_df[
    ["Accuracy", "Precision", "Recall", "F1", "AUC"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Cross-Dataset Baseline on PhiUSIIL Test Set")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(BASELINE_DIR, "baseline_metrics.png"),
    dpi=200
)

plt.close()


# ============================================================
# 15. Save experiment metadata
# ============================================================

metadata = {
    "random_state": RANDOM_STATE,
    "phi_split": {
        "train": 0.75,
        "validation": 0.15,
        "test": 0.10
    },
    "internal_label": {
        "0": "legitimate",
        "1": "phishing"
    },
    "cnn_model": CNN_MODEL_PATH,
    "mlp_model": MLP_MODEL_PATH,
    "mlp_adapter": (
        "URL-observable UCI features reconstructed from URL; "
        "non-URL-observable features filled with UCI training mean."
    ),
    "max_length": MAX_LENGTH,
    "late_fusion": {
        "alpha_cnn": LATE_ALPHA,
        "alpha_mlp": 1.0 - LATE_ALPHA,
        "selection": "fixed_0.5_zero_shot"
    },
    "experiment": "Experiment 1 — Zero-Shot",
    "uses_phiusiil_train_labels": False,
    "uses_phiusiil_validation_labels": False
}

with open(
    os.path.join(BASELINE_DIR, "baseline_config.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)


print("\n" + "=" * 70)
print("01 BASELINE COMPLETE")
print("=" * 70)
print("Results:", results_path)
print("Split files:", BASELINE_DIR)
print("The PhiUSIIL TEST split is now frozen for 02_fusion.py / 03_robustness.py.")