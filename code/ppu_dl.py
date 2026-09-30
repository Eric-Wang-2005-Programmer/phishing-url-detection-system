# ============================================================
# UCI ML Phishing Dataset
# MLP Classification
#
# Dataset Split:
#   75% Training
#   15% Validation
#   10% Test
#
# Model:
#   MLPClassifier
#   Hidden Layers: 64 -> 32
#
# Important:
#   - StandardScaler 只在 Training set 上 fit
#   - Validation / Test 只使用 transform
#   - 不使用 sklearn 內建 early_stopping
#     避免 Training set 再被內部分割
#   - Test set 最後才做正式評估
# ============================================================


# ============================================================
# 1. Import
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. 建立專案資料夾
# ============================================================

BASE_DIR = '/content/drive/MyDrive/phishing_project/uci_mlp'

os.makedirs(BASE_DIR, exist_ok=True)

print("Model / result directory:")
print(BASE_DIR)


# ============================================================
# 3. 讀取 Dataset
# ============================================================

csv_path = '/content/drive/MyDrive/phishing_project/uci_mlp/uci-ml-phishing-dataset.csv'

df = pd.read_csv(csv_path)

print("\n=== Dataset Information ===")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. 檢查 Target Result
# ============================================================

print("\n=== Result Distribution ===")

print(df["Result"].value_counts())

print("\nUnique Result values:")
print(df["Result"].unique())


# ============================================================
# 5. 移除 ID
# 因為ID 本身沒有語意。如果讓模型吃 ID：模型可能把數字關係誤當成特徵。
# ============================================================

if "id" in df.columns:
    df = df.drop(columns=["id"])

    print("\n'id' column removed.")


# ============================================================
# 6. 建立 Features X 與 Label y

# ============================================================
#把資料集裡除了 Result 以外的所有欄位
#（如 having_IP_Address、URL_Length 等 31 個特徵欄位）全部指定給 X。
# 這些是模型用來進行預測的依據。
X = df.drop(columns=["Result"])

# 原始程式：
# y = (df["Result"] == 1).astype(int)

# 保留原本的 label 定義
#0 = Result 不是 1
#1 = Result 是 1
y = (df["Result"] == 1).astype(int)



# ============================================================
# 7. 檢查資料
# ============================================================

print("\n=== Feature Information ===")

print("X shape:", X.shape)

print("y shape:", y.shape)

print("\nNumber of missing values:")
print(X.isnull().sum().sum())

print("\nFeature data types:")
print(X.dtypes.value_counts())


# ============================================================
# 8. 確認所有 Feature 都是數值
# ============================================================

non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if len(non_numeric_columns) > 0:

    print("\nNon-numeric columns found:")
    print(non_numeric_columns)

    raise ValueError(
        "X contains non-numeric features. "
        "Please preprocess them before training."
    )

else:

    print("\nAll features are numeric.")


# ============================================================
# 9. 75% Training / 15% Validation / 10% Test
# ============================================================

# 第一階段：
#
# 75% Training
# 25% Temporary
#
# Temporary = 25%

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 第二階段：
#
# Temporary 25%
#
# 60% of Temporary -> Validation
# 40% of Temporary -> Test
#
# 25% × 60% = 15%
# 25% × 40% = 10%

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.4,
    random_state=42,
    stratify=y_temp #注意，加入這個會盡量讓：Training、Test都保持原本的類別比例。
)


# ============================================================
# 10. 檢查 Dataset Split
# ============================================================

print("\n=== Dataset Split ===")

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

total = len(X)

print("\nActual ratios:")

print(
    "Training:",
    len(X_train) / total
)

print(
    "Validation:",
    len(X_val) / total
)

print(
    "Test:",
    len(X_test) / total
)


# ============================================================
# 11. StandardScaler
# ============================================================
#標準化個特徵，確保不會因為特徵尺度差異而影響訓練
scaler = StandardScaler()


# ------------------------------------------------------------
# IMPORTANT:
#
# scaler 只能在 Training set 上 fit
#
# 不能：
# scaler.fit(X)
#
# 否則 Validation / Test 的資訊會洩漏到 Training
# ------------------------------------------------------------

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


print("\n=== Scaling ===")

print("Training data scaled.")

print("Validation data transformed.")

print("Test data transformed.")


# ============================================================
# 12. 建立 MLP Model
# ============================================================

model = MLPClassifier(

    # 兩層Hidden Layer:
    # 64 neurons -> 32 neurons
    hidden_layer_sizes=(64, 32),

    # Activation function
    activation="relu",

    # Optimizer
    solver="adam",

    # L2 regularization，防止模型太過度依賴某些權重，降低 overfitting。
    alpha=1e-4,

    # Mini-batch size，一次更新的資料筆數
    batch_size=32,

    # Initial learning rate
    learning_rate_init=1e-3,

    # Maximum number of iterations，防止模型太過度依賴某些權重，降低 overfitting。
    max_iter=200,

    # --------------------------------------------------------
    # 不使用 sklearn 內建 early stopping
    #
    # 因為我們已經明確建立：
    #
    # 75% Training
    # 15% Validation
    # 10% Test
    #
    # sklearn 的 early_stopping=True
    # 會再從 Training set 裡切一部分 validation，
    # 與我們現在的設計不同。
    # --------------------------------------------------------

    early_stopping=False,

    # Random seed
    random_state=42
)


# ============================================================
# 13. Training:正式訓練
# ============================================================

print("\n========================================")
print("Start MLP Training")
print("========================================")

model.fit(
    X_train_scaled,
    y_train
)


print("\nTraining finished.")

print("Number of iterations:", model.n_iter_)

print("Final loss:", model.loss_)


# ============================================================
# 14. Validation Evaluation
# ============================================================

print("\n========================================")
print("Validation Evaluation")
print("========================================")


y_val_pred = model.predict(X_val_scaled)


val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)


val_macro_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro"
)


val_weighted_f1 = f1_score(
    y_val,
    y_val_pred,
    average="weighted"
)


print("\nValidation Accuracy:")
print(val_accuracy)


print("\nValidation Macro F1:")
print(val_macro_f1)


print("\nValidation Weighted F1:")
print(val_weighted_f1)


print("\nValidation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)


print("\nValidation Classification Report:")

print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=[
            "Benign",
            "Phishing"
        ]
    )
)


# ============================================================
# 15. Test Evaluation
# ============================================================

print("\n========================================")
print("Final Test Evaluation")
print("========================================")


# Test set 只在這裡正式使用
y_test_pred = model.predict(X_test_scaled)


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


# ------------------------------------------------------------
# Macro F1
# ------------------------------------------------------------

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)


# ------------------------------------------------------------
# Weighted F1
# ------------------------------------------------------------

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)


# ============================================================
# 16. 顯示 Test Results
# ============================================================

print("\n=== Test Results ===")


print("\nAccuracy:")
print(test_accuracy)


print("\nMacro F1:")
print(test_macro_f1)


print("\nWeighted F1:")
print(test_weighted_f1)


print("\nConfusion Matrix:")

test_cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(test_cm)


print("\nClassification Report:")

test_report = classification_report(
    y_test,
    y_test_pred,
    target_names=[
        "Benign",
        "Phishing"
    ]
)

print(test_report)


# ============================================================
# 17. 儲存 MLP Model
# ============================================================

model_path = os.path.join(
    BASE_DIR,
    'uci_mlp_model.pkl'
)

joblib.dump(
    model,
    model_path
)


print("\nMLP model saved to:")

print(model_path)


# ============================================================
# 18. 儲存 StandardScaler
# ============================================================

scaler_path = os.path.join(
    BASE_DIR,
    'uci_scaler.pkl'
)

joblib.dump(
    scaler,
    scaler_path
)


print("\nScaler saved to:")

print(scaler_path)


# ============================================================
# 19. 儲存 Label Mapping
# ============================================================

label_mapping = {
    0: "Benign",
    1: "Phishing"
}


label_mapping_path = os.path.join(
    BASE_DIR,
    'label_mapping.pkl'
)

joblib.dump(
    label_mapping,
    label_mapping_path
)


print("\nLabel mapping saved to:")

print(label_mapping_path)


# ============================================================
# 20. 儲存 Training History
# ============================================================

training_history = {

    "loss_curve": model.loss_curve_,

    "n_iter": model.n_iter_,

    "final_loss": model.loss_,

    "validation_accuracy": val_accuracy,

    "validation_macro_f1": val_macro_f1,

    "validation_weighted_f1": val_weighted_f1,

    "test_accuracy": test_accuracy,

    "test_macro_f1": test_macro_f1,

    "test_weighted_f1": test_weighted_f1

}


history_path = os.path.join(
    BASE_DIR,
    'training_history.pkl'
)

joblib.dump(
    training_history,
    history_path
)


print("\nTraining history saved to:")

print(history_path)


# ============================================================
# 21. 儲存 Training Configuration
# ============================================================

training_config = {

    "dataset": csv_path,

    "split": {
        "train": 0.75,
        "validation": 0.15,
        "test": 0.10
    },

    "random_state": 42,

    "model": {
        "type": "MLPClassifier",
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 32,
        "learning_rate_init": 1e-3,
        "max_iter": 200,
        "early_stopping": False
    },

    "scaler": "StandardScaler",

    "target_definition": "Result == 1 -> 1, otherwise 0"

}


config_path = os.path.join(
    BASE_DIR,
    'training_config.pkl'
)

joblib.dump(
    training_config,
    config_path
)


print("\nTraining configuration saved to:")

print(config_path)


# ============================================================
# 22. 儲存 Test Prediction
# ============================================================

test_results = X_test.copy()

test_results["true_label"] = y_test.values

test_results["predicted_label"] = y_test_pred


prediction_path = os.path.join(
    BASE_DIR,
    'test_predictions.csv'
)

test_results.to_csv(
    prediction_path,
    index=False
)


print("\nTest predictions saved to:")

print(prediction_path)


# ============================================================
# 23. 最終 Summary
# ============================================================

print("\n")
print("========================================")
print("Training Complete")
print("========================================")

print("\nDataset:")
print(csv_path)

print("\nSplit:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Test       :", len(X_test))

print("\nModel:")
print("MLP 64 -> 32")

print("\nValidation:")
print("Accuracy    :", val_accuracy)
print("Macro F1    :", val_macro_f1)

print("\nFinal Test:")
print("Accuracy    :", test_accuracy)
print("Macro F1    :", test_macro_f1)
print("Weighted F1 :", test_weighted_f1)

print("\nSaved Files:")

print("1.", model_path)
print("2.", scaler_path)
print("3.", label_mapping_path)
print("4.", history_path)
print("5.", config_path)
print("6.", prediction_path)

print("\n========================================")
print("All files saved successfully.")
print("========================================")