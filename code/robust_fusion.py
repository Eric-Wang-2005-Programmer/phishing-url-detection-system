import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# 1. 路徑設定 (Paths)
# ============================================================
ROOT = "/content/drive/MyDrive/phishing_project"
FUSION = os.path.join(ROOT, "fusion")
OUT = os.path.join(ROOT, "robust_fusion")
os.makedirs(OUT, exist_ok=True)

R = os.path.join(FUSION, "representations")

# ============================================================
# 2. 載入資料 (Load Data)
# ============================================================
cnn_old = np.load(os.path.join(R, "cnn_embeddings.npy"))
mlp = np.load(os.path.join(R, "mlp_hidden.npy"))
y = np.load(os.path.join(R, "labels.npy"))

train = np.load(os.path.join(R, "original_train_idx.npy"))
val = np.load(os.path.join(R, "original_val_idx.npy"))
test = np.load(os.path.join(R, "original_test_idx.npy"))

meta = pd.read_csv(os.path.join(R, "phiusiil_metadata.csv"))
urls = meta["URL"].astype(str).values

test_urls = urls[test]

print("Original stored CNN shape:", cnn_old.shape)
print("MLP shape:", mlp.shape)

# ============================================================
# 3. 載入基礎 CNN 模型與 Tokenizer
# ============================================================
cnn_model = load_model(os.path.join(ROOT, "cnn", "malicious_url_cnn.keras"))

with open(os.path.join(ROOT, "cnn", "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

def cnn_embed(urls_input):
    x = pad_sequences(
        tokenizer.texts_to_sequences(urls_input),
        maxlen=150,
        padding="post",
        truncating="post"
    )
    return cnn_model.layers[-2](x).numpy()

# ============================================================
# 4. 定義 6 種干擾函數 (Perturbation Functions)
# ============================================================
def slash(u):
    return u.rstrip("/") + "/"

def dot(u):
    return u.rstrip("/") + "."

def hyphen_query(u):
    if "?" in u:
        return u.replace("?", "-?")
    return u.rstrip("/") + "-"

def add_query(u):
    return u + ("&" if "?" in u else "?") + "q=test"

def change_case(u):
    return u.swapcase()

def make_aug(urls_input):
    return np.concatenate([
        urls_input,
        np.array([slash(u) for u in urls_input]),
        np.array([dot(u) for u in urls_input])
    ])

# ============================================================
# 5. 建立對抗式訓練與驗證資料 (Robust Training & Val Data)
# ============================================================
print("\nGenerating augmented CNN embeddings for Training...")
train_urls = urls[train]
aug_urls = make_aug(train_urls)
cnn_aug = cnn_embed(aug_urls)

mlp_aug = np.tile(mlp[train], (3, 1))
H_train = np.concatenate([cnn_aug, mlp_aug], axis=1)
y_train = np.tile(y[train], 3)

print("H_train shape:", H_train.shape)

print("\nGenerating validation CNN embeddings...")
val_cnn = cnn_embed(urls[val])
H_val = np.concatenate([val_cnn, mlp[val]], axis=1)
print("H_val shape:", H_val.shape)

# ============================================================
# 6. 建構與訓練 Robust Intermediate Fusion 模型
# ============================================================
model = tf.keras.Sequential([
    Input(shape=(H_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

print("\nTraining Robust Intermediate Fusion Model...")
model.fit(
    H_train,
    y_train,
    validation_data=(H_val, y[val]),
    epochs=30,
    batch_size=256,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# 儲存訓練好的模型
model_save_path = os.path.join(OUT, "robust_intermediate_fusion.keras")
model.save(model_save_path)
print(f"\nModel successfully saved to: {model_save_path}")

# ============================================================
# 7. 全面壓力測試 (Comprehensive Robustness Evaluation - All 6)
# ============================================================
def evaluate(name, H):
    p = model.predict(H, batch_size=512, verbose=0).ravel()
    pred = (p >= 0.5).astype(int)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y[test], pred),
        "Precision": precision_score(y[test], pred, zero_division=0),
        "Recall": recall_score(y[test], pred, zero_division=0),
        "F1": f1_score(y[test], pred, zero_division=0),
        "AUC": roc_auc_score(y[test], p)
    }

    print(
        f"{name:16s}"
        f" Acc={result['Accuracy']:.4f}"
        f" Prec={result['Precision']:.4f}"
        f" Recall={result['Recall']:.4f}"
        f" F1={result['F1']:.4f}"
        f" AUC={result['AUC']:.4f}"
    )

    return result

PERTURBATIONS = {
    "Original": lambda u: u,
    "AddSlash": slash,
    "AddDot": dot,
    "AddHyphenQuery": hyphen_query,
    "AddQuery": add_query,
    "ChangeCase": change_case
}

print("\n========================================")
print("Evaluating ALL 6 Perturbations on Test Set")
print("========================================")

results = []
for name, func in PERTURBATIONS.items():
    print(f"Testing {name}...")
    perturbed = np.array([func(u) for u in test_urls], dtype=str)
    cnn_rep = cnn_embed(perturbed)
    H = np.concatenate([cnn_rep, mlp[test]], axis=1)
    results.append(evaluate(name, H))

# ============================================================
# 8. 儲存全面測試結果 (Save Results)
# ============================================================
results_df = pd.DataFrame(results)
csv_save_path = os.path.join(OUT, "robust_fusion_all_perturbations.csv")
results_df.to_csv(csv_save_path, index=False)

print("\n========================================")
print("ALL ROBUSTNESS RESULTS")
print("========================================")
print(results_df.to_string(index=False))
print(f"\nSaved CSV to: {csv_save_path}")