import os, json, pickle, random, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlsplit
from ipaddress import ip_address
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer

# ============================================================
# Experiment 1: Target-domain adaptation CNN / MLP / Intermediate
# Experiment 2: CNN + MLP from scratch
# Experiment 3: Data efficiency
# ============================================================

DRIVE_ROOT = "/content/drive/MyDrive"
PROJECT_ROOT = os.path.join(DRIVE_ROOT, "phishing_project")
CNN_DIR = os.path.join(PROJECT_ROOT, "cnn")
MLP_DIR = os.path.join(PROJECT_ROOT, "uci_mlp")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "baseline")
FUSION_DIR = os.path.join(PROJECT_ROOT, "fusion")
UCI_PATH = os.path.join(MLP_DIR, "uci-ml-phishing-dataset.csv")
PHIUSIIL_PATH = os.path.join(PROJECT_ROOT, "PhiUSIIL_Phishing_URL_Dataset.csv")
EFFICIENCY_DIR = os.path.join(PROJECT_ROOT, "data_efficiency")
GROUPED_DIR = os.path.join(FUSION_DIR, "grouped_sensitivity")
for d in [FUSION_DIR, EFFICIENCY_DIR, GROUPED_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
MAX_LENGTH = 150
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
EPOCHS = 30
BATCH = 256
PATIENCE = 5

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

def evaluate_binary(y_true, prob):
    prob = np.asarray(prob).reshape(-1)
    pred = (prob >= 0.5).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["AUC"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["AUC"] = np.nan
    return out

def norm_col(x):
    return str(x).strip().lower().replace(" ", "").replace("-", "_")

# ============================================================
# UCI-compatible URL adapter
# ============================================================
SHORTENING_SERVICES = {
    "bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","is.gd","buff.ly",
    "adf.ly","bit.do","cutt.ly","tiny.cc","lnkd.in","rb.gy",
    "shorturl.at","rebrand.ly","shorte.st","cli.gs","bc.vc"
}

def safe_parts(url):
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url
    try:
        p = urlsplit(url)
        return (p.scheme or "").lower(), (p.hostname or "").lower(), p.path or ""
    except Exception:
        return "", "", ""

def is_ip_host(host):
    if not host:
        return False
    try:
        ip_address(host)
        return True
    except Exception:
        return False

def subdomain_count(host):
    return max(0, len([x for x in host.split(".") if x]) - 2)

def url_feature_values(url):
    url = str(url)
    scheme, host, path = safe_parts(url)
    n = len(url)
    url_len = 1 if n < 54 else (0 if n <= 75 else -1)
    subs = subdomain_count(host)
    sub = 1 if subs == 0 else (0 if subs == 1 else -1)
    rest = url.split("://", 1)[1] if "://" in url else url
    return {
        "having_ip_address": -1 if is_ip_host(host) else 1,
        "url_length": url_len,
        "shortining_service": -1 if host in SHORTENING_SERVICES else 1,
        "having_at_symbol": -1 if "@" in url else 1,
        "double_slash_redirecting": -1 if "//" in rest else 1,
        "prefix_suffix": -1 if "-" in host else 1,
        "having_sub_domain": sub,
        "https_token": -1 if "https" in host else 1,
        "abnormal_url": 1, "redirect": 1, "submitting_to_email": 1,
        "on_mouseover": 1, "rightclick": 1, "popupwidnow": 1, "iframe": 1
    }

def build_uci_adapter(urls, columns, scaler, reference_df):
    means = reference_df[columns].mean(numeric_only=True)
    rows = []
    for url in urls:
        uv = url_feature_values(url)
        rows.append({
            c: uv[norm_col(c)] if norm_col(c) in uv else means[c]
            for c in columns
        })
    X = pd.DataFrame(rows, columns=columns)
    if X.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Expected {scaler.n_features_in_} MLP features, got {X.shape[1]}.")
    return X

# ============================================================
# Load data/models & extract representations
# ============================================================
print("="*75)
print("LOAD MODELS / DATA & EXTRACT REPRESENTATIONS")
print("="*75)

cnn = load_model(os.path.join(CNN_DIR, "malicious_url_cnn.keras"))
with open(os.path.join(CNN_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
with open(os.path.join(CNN_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
mlp = joblib.load(os.path.join(MLP_DIR, "uci_mlp_model.pkl"))
scaler = joblib.load(os.path.join(MLP_DIR, "uci_scaler.pkl"))

phi = pd.read_csv(PHIUSIIL_PATH)
label_candidates = [c for c in phi.columns if norm_col(c) == "label"]
if not label_candidates:
    raise ValueError("PhiUSIIL label column not found.")
if "URL" not in phi.columns:
    raise ValueError("PhiUSIIL URL column 'URL' not found.")

urls = phi["URL"].astype(str).values
y = (phi[label_candidates[0]].astype(int) == 0).astype(int).values

train_idx = np.load(os.path.join(BASELINE_DIR, "phi_train_idx.npy"))
val_idx = np.load(os.path.join(BASELINE_DIR, "phi_val_idx.npy"))
test_idx = np.load(os.path.join(BASELINE_DIR, "phi_test_idx.npy"))

def tokenize_urls(arr):
    seq = tokenizer.texts_to_sequences(pd.Series(arr).astype(str))
    return tf.keras.utils.pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")

cnn_input = Input(shape=(MAX_LENGTH,), dtype="int32")
x = cnn_input
pool_output = None
for layer in cnn.layers:
    if layer.__class__.__name__ == "InputLayer":
        continue
    x = layer(x)
    if layer.__class__.__name__ == "GlobalMaxPooling1D":
        pool_output = x
cnn_embedding_model = Model(cnn_input, pool_output, name="cnn_embedding_extractor")

cnn_emb_all = cnn_embedding_model.predict(tokenize_urls(urls), batch_size=512, verbose=0)

uci_df = pd.read_csv(UCI_PATH)
if "id" in uci_df.columns:
    uci_df = uci_df.drop(columns=["id"])
uci_columns = [c for c in uci_df.columns if c != "Result"]
reference_df, _ = train_test_split(
    uci_df, test_size=0.25, random_state=RANDOM_STATE, stratify=uci_df["Result"]
)
X_uci = build_uci_adapter(urls, uci_columns, scaler, reference_df)
X_uci_scaled = scaler.transform(X_uci)

def relu(x):
    return np.maximum(x, 0.0)

def sklearn_mlp_hidden(model, X):
    A = X
    for i in range(len(model.coefs_) - 1):
        A = A @ model.coefs_[i] + model.intercepts_[i]
        A = relu(A)
    return A

mlp_hidden = sklearn_mlp_hidden(mlp, X_uci_scaled)
H = np.concatenate([cnn_emb_all, mlp_hidden], axis=1)

# Save representations
REP_DIR = os.path.join(FUSION_DIR, "representations")
os.makedirs(REP_DIR, exist_ok=True)
np.save(os.path.join(REP_DIR, "cnn_embeddings.npy"), cnn_emb_all)
np.save(os.path.join(REP_DIR, "mlp_hidden.npy"), mlp_hidden)
np.save(os.path.join(REP_DIR, "fusion_H.npy"), H)
np.save(os.path.join(REP_DIR, "labels.npy"), y)
np.save(os.path.join(REP_DIR, "original_train_idx.npy"), train_idx)
np.save(os.path.join(REP_DIR, "original_val_idx.npy"), val_idx)
np.save(os.path.join(REP_DIR, "original_test_idx.npy"), test_idx)
pd.DataFrame({"row_index": np.arange(len(urls)), "URL": urls, "label": y}).to_csv(
    os.path.join(REP_DIR, "phiusiil_metadata.csv"), index=False
)

# ============================================================
# Adaptation head helpers
# ============================================================
def build_head(input_dim, name):
    inp = Input(shape=(input_dim,), name=name+"_input")
    x = Dense(64, activation="relu", name=name+"_dense64")(inp)
    x = Dropout(0.30, name=name+"_dropout")(x)
    out = Dense(1, activation="sigmoid", name=name+"_output")(x)
    model = Model(inp, out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def train_head(Xtr, ytr, Xv, yv, dim, name, save_path, verbose=1):
    model = build_head(dim, name)
    cb = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True)
    ]
    model.fit(
        Xtr, ytr, validation_data=(Xv, yv),
        epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=verbose
    )
    return load_model(save_path)

# ============================================================
# Experiment 1 — Target-domain adaptation
# ============================================================
print("\n"+"="*75)
print("EXPERIMENT 1 — TARGET-DOMAIN ADAPTATION")
print("="*75)

cnn_model = train_head(
    cnn_emb_all[train_idx], y[train_idx], cnn_emb_all[val_idx], y[val_idx],
    cnn_emb_all.shape[1], "cnn_adaptation_head", os.path.join(FUSION_DIR, "cnn_adaptation_head.keras")
)
mlp_model = train_head(
    mlp_hidden[train_idx], y[train_idx], mlp_hidden[val_idx], y[val_idx],
    mlp_hidden.shape[1], "mlp_adaptation_head", os.path.join(FUSION_DIR, "mlp_adaptation_head.keras")
)
fusion_model = train_head(
    H[train_idx], y[train_idx], H[val_idx], y[val_idx],
    H.shape[1], "intermediate_fusion_adaptation", os.path.join(FUSION_DIR, "intermediate_fusion_adaptation.keras")
)

p_cnn_adapt = cnn_model.predict(cnn_emb_all, batch_size=512, verbose=0).reshape(-1)
p_mlp_adapt = mlp_model.predict(mlp_hidden, batch_size=512, verbose=0).reshape(-1)
p_intermediate = fusion_model.predict(H, batch_size=512, verbose=0).reshape(-1)

exp2_models = {
    "CNN_Adaptation": p_cnn_adapt,
    "MLP_Adaptation": p_mlp_adapt,
    "Intermediate_Fusion": p_intermediate
}
exp2_rows = [{"Experiment": "Experiment 2", "Model": name, **evaluate_binary(y[test_idx], p[test_idx])} for name, p in exp2_models.items()]
exp2 = pd.DataFrame(exp2_rows)
exp2.to_csv(os.path.join(FUSION_DIR, "experiment2_results.csv"), index=False)

pd.DataFrame({
    "row_index": test_idx, "URL": urls[test_idx], "true_label": y[test_idx],
    **{k: v[test_idx] for k, v in exp2_models.items()}
}).to_csv(os.path.join(FUSION_DIR, "experiment2_test_predictions.csv"), index=False)

for name, p in exp2_models.items():
    np.save(os.path.join(FUSION_DIR, name.lower()+"_probability.npy"), p)

print(exp2.to_string(index=False))

# ============================================================
# Experiment 2 — From scratch
# ============================================================
print("\n"+"="*75)
print("EXPERIMENT 2 — FROM SCRATCH")
print("="*75)

scratch_tokenizer = Tokenizer(char_level=True, filters="", lower=False, oov_token="<OOV>")
scratch_tokenizer.fit_on_texts(pd.Series(urls[train_idx]).astype(str))

def scratch_tokens(arr):
    return tf.keras.utils.pad_sequences(
        scratch_tokenizer.texts_to_sequences(pd.Series(arr).astype(str)),
        maxlen=MAX_LENGTH, padding="post", truncating="post"
    )

scratch_cnn = tf.keras.Sequential([
    Input(shape=(MAX_LENGTH,), dtype="int32"),
    tf.keras.layers.Embedding(len(scratch_tokenizer.word_index)+1, 32),
    tf.keras.layers.Conv1D(128, 5, activation="relu"),
    tf.keras.layers.GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.50),
    Dense(1, activation="sigmoid")
], name="cnn_from_scratch")
scratch_cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
scratch_path = os.path.join(FUSION_DIR, "cnn_from_scratch.keras")
scratch_cnn.fit(
    scratch_tokens(urls[train_idx]), y[train_idx],
    validation_data=(scratch_tokens(urls[val_idx]), y[val_idx]),
    epochs=EPOCHS, batch_size=512,
    callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True), ModelCheckpoint(scratch_path, monitor="val_loss", save_best_only=True)],
    verbose=1
)
scratch_cnn = load_model(scratch_path)
p_scratch = scratch_cnn.predict(scratch_tokens(urls), batch_size=512, verbose=0).reshape(-1)

mlp_scratch = tf.keras.Sequential([
    Input(shape=(X_uci_scaled.shape[1],), dtype="float32"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dropout(0.30),
    Dense(1, activation="sigmoid")
], name="mlp_from_scratch")
mlp_scratch.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
mlp_scratch_path = os.path.join(FUSION_DIR, "mlp_from_scratch.keras")
mlp_scratch.fit(
    X_uci_scaled[train_idx].astype(np.float32), y[train_idx],
    validation_data=(X_uci_scaled[val_idx].astype(np.float32), y[val_idx]),
    epochs=EPOCHS, batch_size=256,
    callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True), ModelCheckpoint(mlp_scratch_path, monitor="val_loss", save_best_only=True)],
    verbose=1
)
mlp_scratch = load_model(mlp_scratch_path)
p_mlp_scratch = mlp_scratch.predict(X_uci_scaled.astype(np.float32), batch_size=512, verbose=0).reshape(-1)

exp3 = pd.DataFrame([
    {"Experiment": "Experiment 3", "Model": "CNN_From_Scratch", **evaluate_binary(y[test_idx], p_scratch[test_idx])},
    {"Experiment": "Experiment 3", "Model": "MLP_From_Scratch", **evaluate_binary(y[test_idx], p_mlp_scratch[test_idx])}
])
exp3.to_csv(os.path.join(FUSION_DIR, "experiment3_results.csv"), index=False)
np.save(os.path.join(FUSION_DIR, "cnn_from_scratch_probability.npy"), p_scratch)
np.save(os.path.join(FUSION_DIR, "mlp_from_scratch_probability.npy"), p_mlp_scratch)
with open(os.path.join(FUSION_DIR, "scratch_tokenizer.pkl"), "wb") as f:
    pickle.dump(scratch_tokenizer, f)
print(exp3.to_string(index=False))

# ============================================================
# Experiment 3 — Data efficiency
# ============================================================
print("\n"+"="*75)
print("EXPERIMENT 3 — DATA EFFICIENCY")
print("="*75)

eff_rows = []
for frac in FRACTIONS:
    percent = int(frac * 100)
    print(f"Target train fraction: {percent}%")
    frac_idx, _ = train_test_split(
        train_idx, test_size=1-frac,
        random_state=RANDOM_STATE+percent,
        stratify=y[train_idx]
    )

    for model_name, Xall in [
        ("CNN_Adaptation", cnn_emb_all),
        ("MLP_Adaptation", mlp_hidden),
        ("Intermediate_Fusion", H)
    ]:
        tf.keras.backend.clear_session()
        dim = Xall.shape[1]
        model = build_head(dim, f"eff_{model_name}_{percent}")
        model.fit(
            Xall[frac_idx], y[frac_idx],
            validation_data=(Xall[val_idx], y[val_idx]),
            epochs=EPOCHS, batch_size=BATCH,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
            verbose=0
        )
        p = model.predict(Xall[test_idx], batch_size=512, verbose=0).reshape(-1)
        eff_rows.append({
            "Experiment": "Experiment 4", "Fraction": frac, "Percent": percent,
            "Train_Samples": len(frac_idx), "Model": model_name,
            **evaluate_binary(y[test_idx], p)
        })

eff = pd.DataFrame(eff_rows)
eff.to_csv(os.path.join(EFFICIENCY_DIR, "experiment4_data_efficiency.csv"), index=False)

for metric in ["F1", "AUC"]:
    plt.figure(figsize=(10, 6))
    for model_name in ["CNN_Adaptation", "MLP_Adaptation", "Intermediate_Fusion"]:
        d = eff[eff["Model"] == model_name].sort_values("Percent")
        plt.plot(d["Percent"], d[metric], marker="o", label=model_name)
    plt.xlabel("Target-domain training data (%)")
    plt.ylabel(metric)
    plt.title(f"Experiment 4 — Data Efficiency ({metric})")
    plt.xticks([1, 5, 10, 25, 50, 75])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EFFICIENCY_DIR, f"data_efficiency_{metric}.png"), dpi=200)
    plt.close()

print(eff.to_string(index=False))


