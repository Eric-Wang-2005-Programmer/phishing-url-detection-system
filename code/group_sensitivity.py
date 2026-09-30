# ============================================================
# Grouped URL Split Sensitivity Analysis
#
# Purpose:
#   Check whether random train/validation/test splitting
#   produces overly optimistic results because identical
#   normalized URLs appear across different splits.
#
# Split:
#   75% Train / 15% Validation / 10% Test
#
# Models:
#   1. CNN Adaptation
#   2. MLP Adaptation
#   3. Intermediate Fusion
#
# Comparison:
#   Random Split vs Grouped Split
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


warnings.filterwarnings("ignore")


# ============================================================
# 2. Paths
# ============================================================

DRIVE_ROOT = "/content/drive/MyDrive"

PROJECT_ROOT = os.path.join(
    DRIVE_ROOT,
    "phishing_project"
)

FUSION_DIR = os.path.join(
    PROJECT_ROOT,
    "fusion"
)

REPRESENTATION_DIR = os.path.join(
    FUSION_DIR,
    "representations"
)

GROUPED_DIR = os.path.join(
    FUSION_DIR,
    "grouped_sensitivity"
)

os.makedirs(
    GROUPED_DIR,
    exist_ok=True
)


# ============================================================
# 3. Reproducibility
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 4. Configuration
# ============================================================

EPOCHS = 30
BATCH_SIZE = 256
PATIENCE = 5


# ============================================================
# 5. Helper: Binary Evaluation
# ============================================================

def evaluate_binary(y_true, prob):

    pred = (
        prob >= 0.5
    ).astype(int)

    result = {
        "Accuracy":
            accuracy_score(
                y_true,
                pred
            ),

        "Precision":
            precision_score(
                y_true,
                pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                pred,
                zero_division=0
            )
    }

    try:

        result["AUC"] = roc_auc_score(
            y_true,
            prob
        )

    except ValueError:

        result["AUC"] = np.nan

    return result


# ============================================================
# 6. Fast URL Normalization
# ============================================================

def normalize_url_group_fast(url):

    """
    Conservative URL normalization.

    Operations:
      1. strip whitespace
      2. lowercase
      3. remove fragment (#...)
      4. add http:// if scheme is missing

    URLs are considered the same group only when
    their normalized strings are exactly identical.
    """

    s = str(url).strip().lower()

    # Remove fragment
    if "#" in s:

        s = s.split(
            "#",
            1
        )[0]

    # Add scheme when missing
    if "://" not in s:

        s = "http://" + s

    return s


# ============================================================
# 7. Load Representations
# ============================================================

print("=" * 70)
print("GROUPED URL SPLIT SENSITIVITY ANALYSIS")
print("=" * 70)

print()
print("Loading representations...")
print()


cnn_embeddings = np.load(
    os.path.join(
        REPRESENTATION_DIR,
        "cnn_embeddings.npy"
    ),
    mmap_mode="r"
)

mlp_hidden = np.load(
    os.path.join(
        REPRESENTATION_DIR,
        "mlp_hidden.npy"
    ),
    mmap_mode="r"
)

fusion_H = np.load(
    os.path.join(
        REPRESENTATION_DIR,
        "fusion_H.npy"
    ),
    mmap_mode="r"
)

labels = np.load(
    os.path.join(
        REPRESENTATION_DIR,
        "labels.npy"
    ),
    mmap_mode="r"
)

metadata = pd.read_csv(
    os.path.join(
        REPRESENTATION_DIR,
        "phiusiil_metadata.csv"
    )
)


print("Representation shapes:")
print(
    "CNN embedding:",
    cnn_embeddings.shape
)

print(
    "MLP hidden:",
    mlp_hidden.shape
)

print(
    "Fusion H:",
    fusion_H.shape
)

print(
    "Labels:",
    labels.shape
)

print(
    "Metadata:",
    metadata.shape
)


# ============================================================
# 8. Verify URL Column
# ============================================================

print()
print("=" * 70)
print("CHECKING URL COLUMN")
print("=" * 70)

print()

print(
    "Metadata columns:",
    list(metadata.columns)
)

if "URL" not in metadata.columns:

    raise ValueError(
        "Cannot find 'URL' column in phiusiil_metadata.csv"
    )


urls = metadata["URL"]


# ============================================================
# 9. Create Normalized URL Groups
# ============================================================

print()
print("=" * 70)
print("CREATING NORMALIZED URL GROUPS")
print("=" * 70)

print()
print("Normalizing URLs...")
print()


# ------------------------------------------------------------
# IMPORTANT:
# Use pandas vectorized string operations instead of
# Python urlsplit() for every row.
# ------------------------------------------------------------

url_series = (
    metadata["URL"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Remove URL fragments
url_series = (
    url_series
    .str.split(
        "#",
        n=1
    )
    .str[0]
)


# Detect URLs without a scheme
mask_no_scheme = ~url_series.str.contains(
    r"://",
    regex=True,
    na=False
)


# Add http:// only where necessary
url_series.loc[mask_no_scheme] = (
    "http://" +
    url_series.loc[mask_no_scheme]
)


groups = url_series.to_numpy()


# ============================================================
# 10. Group Statistics
# ============================================================

num_rows = len(groups)

unique_groups = np.unique(
    groups
)

num_unique_groups = len(
    unique_groups
)

num_duplicate_rows = (
    num_rows -
    num_unique_groups
)


print(
    "Normalized URL groups created."
)

print(
    "Total rows:",
    num_rows
)

print(
    "Unique groups:",
    num_unique_groups
)

print(
    "Duplicate rows:",
    num_duplicate_rows
)

if num_rows > 0:

    duplicate_rate = (
        num_duplicate_rows /
        num_rows
    )

else:

    duplicate_rate = 0.0


print(
    "Duplicate rate:",
    f"{duplicate_rate:.4%}"
)


# ============================================================
# 11. Save Normalized Groups
# ============================================================

group_table = pd.DataFrame({

    "row_index":
        np.arange(num_rows),

    "URL":
        metadata["URL"].astype(str),

    "normalized_URL":
        groups,

    "label":
        np.asarray(labels)

})


group_table.to_csv(
    os.path.join(
        GROUPED_DIR,
        "normalized_url_groups.csv"
    ),
    index=False
)


# ============================================================
# 12. Create Grouped Train / Validation / Test Split
# ============================================================

print()
print("=" * 70)
print("CREATING GROUPED TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print()


all_indices = np.arange(
    num_rows
)


# ------------------------------------------------------------
# First split:
#
# Train = 75%
# Temporary = 25%
# ------------------------------------------------------------

gss_1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.75,
    random_state=RANDOM_STATE
)


train_idx, temp_idx = next(
    gss_1.split(
        all_indices,
        labels,
        groups=groups
    )
)


# ------------------------------------------------------------
# Second split:
#
# Validation = 60% of temporary
# Test       = 40% of temporary
#
# Therefore:
#
# Validation = 25% × 60% = 15%
# Test       = 25% × 40% = 10%
# ------------------------------------------------------------

gss_2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.60,
    random_state=RANDOM_STATE
)


temp_relative = np.arange(
    len(temp_idx)
)


val_relative, test_relative = next(
    gss_2.split(
        temp_relative,
        labels[temp_idx],
        groups=groups[temp_idx]
    )
)


val_idx = temp_idx[
    val_relative
]

test_idx = temp_idx[
    test_relative
]


# ============================================================
# 13. Verify Split Sizes
# ============================================================

print(
    "Train:",
    len(train_idx),
    f"({len(train_idx) / num_rows:.2%})"
)

print(
    "Validation:",
    len(val_idx),
    f"({len(val_idx) / num_rows:.2%})"
)

print(
    "Test:",
    len(test_idx),
    f"({len(test_idx) / num_rows:.2%})"
)


# ============================================================
# 14. Verify No Group Leakage
# ============================================================

train_groups = set(
    groups[train_idx]
)

val_groups = set(
    groups[val_idx]
)

test_groups = set(
    groups[test_idx]
)


train_val_overlap = (
    train_groups &
    val_groups
)

train_test_overlap = (
    train_groups &
    test_groups
)

val_test_overlap = (
    val_groups &
    test_groups
)


print()
print("=" * 70)
print("GROUP LEAKAGE CHECK")
print("=" * 70)

print()

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) != 0
    or
    len(train_test_overlap) != 0
    or
    len(val_test_overlap) != 0
):

    raise RuntimeError(
        "GROUP LEAKAGE DETECTED!"
    )


print()
print(
    "No normalized URL group leakage detected."
)


# ============================================================
# 15. Save Split Indices
# ============================================================

np.save(
    os.path.join(
        GROUPED_DIR,
        "grouped_train_idx.npy"
    ),
    train_idx
)

np.save(
    os.path.join(
        GROUPED_DIR,
        "grouped_val_idx.npy"
    ),
    val_idx
)

np.save(
    os.path.join(
        GROUPED_DIR,
        "grouped_test_idx.npy"
    ),
    test_idx
)


# ============================================================
# 16. Save Split Assignment
# ============================================================

split_assignment = np.empty(
    num_rows,
    dtype=object
)

split_assignment[
    train_idx
] = "train"

split_assignment[
    val_idx
] = "validation"

split_assignment[
    test_idx
] = "test"


group_table[
    "split"
] = split_assignment


group_table.to_csv(
    os.path.join(
        GROUPED_DIR,
        "grouped_split_assignment.csv"
    ),
    index=False
)


# ============================================================
# 17. Label Distribution
# ============================================================

print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print()


for name, idx in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:

    unique, counts = np.unique(
        labels[idx],
        return_counts=True
    )

    print(
        name,
        dict(
            zip(
                unique.tolist(),
                counts.tolist()
            )
        )
    )


# ============================================================
# 18. Model Builder
# ============================================================

def build_adaptation_head(
    input_dim,
    model_name
):

    inputs = Input(
        shape=(input_dim,),
        name=f"{model_name}_input"
    )

    x = Dense(
        64,
        activation="relu",
        name=f"{model_name}_dense64"
    )(inputs)

    x = Dropout(
        0.30,
        name=f"{model_name}_dropout"
    )(x)

    outputs = Dense(
        1,
        activation="sigmoid",
        name=f"{model_name}_output"
    )(x)

    model = Model(
        inputs,
        outputs,
        name=model_name
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model


# ============================================================
# 19. Train One Grouped Model
# ============================================================

def train_grouped_model(
    X,
    y,
    model_name
):

    print()
    print("=" * 70)
    print(
        f"TRAINING: {model_name}"
    )
    print("=" * 70)

    print()

    model = build_adaptation_head(
        input_dim=X.shape[1],
        model_name=model_name
    )


    model_path = os.path.join(
        GROUPED_DIR,
        f"{model_name}.keras"
    )


    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]


    history = model.fit(

        X[train_idx],
        y[train_idx],

        validation_data=(
            X[val_idx],
            y[val_idx]
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_prob = (
        model.predict(
            X[test_idx],
            batch_size=BATCH_SIZE,
            verbose=1
        )
        .reshape(-1)
    )


    metrics = evaluate_binary(
        y[test_idx],
        test_prob
    )


    metrics["Model"] = model_name

    metrics["Train_Size"] = len(
        train_idx
    )

    metrics["Validation_Size"] = len(
        val_idx
    )

    metrics["Test_Size"] = len(
        test_idx
    )


    print()
    print(
        f"{model_name} results:"
    )

    for key, value in metrics.items():

        if isinstance(
            value,
            (float, np.floating)
        ):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )


    return (
        model,
        history,
        test_prob,
        metrics
    )


# ============================================================
# 20. Prepare Labels
# ============================================================

y = np.asarray(
    labels
).astype(np.int32)


# ============================================================
# 21. Experiment A
#    CNN Grouped Adaptation
# ============================================================

cnn_model, cnn_history, cnn_prob, cnn_result = \
    train_grouped_model(

        cnn_embeddings,
        y,

        "cnn_grouped_adaptation"
    )


# ============================================================
# 22. Experiment B
#    MLP Grouped Adaptation
# ============================================================

mlp_model, mlp_history, mlp_prob, mlp_result = \
    train_grouped_model(

        mlp_hidden,
        y,

        "mlp_grouped_adaptation"
    )


# ============================================================
# 23. Experiment C
#    Intermediate Fusion Grouped Adaptation
# ============================================================

fusion_model, fusion_history, fusion_prob, fusion_result = \
    train_grouped_model(

        fusion_H,
        y,

        "fusion_grouped_adaptation"
    )


# ============================================================
# 24. Save Results
# ============================================================

results = pd.DataFrame([

    cnn_result,

    mlp_result,

    fusion_result

])


# Reorder columns

result_columns = [

    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC",
    "Train_Size",
    "Validation_Size",
    "Test_Size"

]


results = results[
    result_columns
]


results_path = os.path.join(
    GROUPED_DIR,
    "grouped_sensitivity_results.csv"
)


results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 25. Save Test Predictions
# ============================================================

test_predictions = pd.DataFrame({

    "row_index":
        test_idx,

    "URL":
        metadata.iloc[
            test_idx
        ]["URL"].values,

    "normalized_URL":
        groups[test_idx],

    "true_label":
        y[test_idx],

    "cnn_probability":
        cnn_prob,

    "mlp_probability":
        mlp_prob,

    "fusion_probability":
        fusion_prob

})


test_predictions[
    "cnn_prediction"
] = (
    test_predictions[
        "cnn_probability"
    ] >= 0.5
).astype(int)


test_predictions[
    "mlp_prediction"
] = (
    test_predictions[
        "mlp_probability"
    ] >= 0.5
).astype(int)


test_predictions[
    "fusion_prediction"
] = (
    test_predictions[
        "fusion_probability"
    ] >= 0.5
).astype(int)


test_predictions.to_csv(
    os.path.join(
        GROUPED_DIR,
        "grouped_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 26. Random vs Grouped Comparison
# ============================================================

random_results_path = os.path.join(
    FUSION_DIR,
    "experiment2_results.csv"
)


if os.path.exists(
    random_results_path
):

    random_results = pd.read_csv(
        random_results_path
    )


    grouped_compare = results[
        [
            "Model",
            "F1",
            "AUC"
        ]
    ].copy()


    # --------------------------------------------------------
    # Normalize model names for comparison
    # --------------------------------------------------------

    def clean_model_name(name):

        s = str(name).lower()

        if "cnn" in s:

            return "CNN"

        if "mlp" in s:

            return "MLP"

        if "fusion" in s:

            return "Intermediate Fusion"

        return str(name)


    grouped_compare[
        "Model_Normalized"
    ] = grouped_compare[
        "Model"
    ].apply(
        clean_model_name
    )


    random_compare = random_results.copy()


    if "Model" in random_compare.columns:

        random_compare[
            "Model_Normalized"
        ] = random_compare[
            "Model"
        ].apply(
            clean_model_name
        )


        random_compare = random_compare[
            [
                "Model_Normalized",
                "F1",
                "AUC"
            ]
        ]


        random_compare = (
            random_compare
            .rename(
                columns={
                    "F1":
                        "Random_F1",

                    "AUC":
                        "Random_AUC"
                }
            )
        )


        grouped_compare = (
            grouped_compare
            .rename(
                columns={
                    "F1":
                        "Grouped_F1",

                    "AUC":
                        "Grouped_AUC"
                }
            )
        )


        comparison = grouped_compare.merge(
            random_compare,
            on="Model_Normalized",
            how="left"
        )


        comparison[
            "F1_Difference"
        ] = (
            comparison["Grouped_F1"]
            -
            comparison["Random_F1"]
        )


        comparison[
            "AUC_Difference"
        ] = (
            comparison["Grouped_AUC"]
            -
            comparison["Random_AUC"]
        )


        comparison_path = os.path.join(
            GROUPED_DIR,
            "random_vs_grouped_comparison.csv"
        )


        comparison.to_csv(
            comparison_path,
            index=False
        )


        print()
        print("=" * 70)
        print("RANDOM VS GROUPED COMPARISON")
        print("=" * 70)
        print()

        print(
            comparison.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # F1 Plot
        # ----------------------------------------------------

        plot_df = comparison[
            [
                "Model_Normalized",
                "Random_F1",
                "Grouped_F1"
            ]
        ].copy()


        plot_df = plot_df.set_index(
            "Model_Normalized"
        )


        ax = plot_df.plot(
            kind="bar",
            figsize=(10, 6)
        )


        ax.set_ylim(
            0,
            1
        )

        ax.set_ylabel(
            "F1 Score"
        )

        ax.set_title(
            "Random Split vs Grouped Split"
        )

        plt.xticks(
            rotation=0
        )

        plt.tight_layout()


        plt.savefig(
            os.path.join(
                GROUPED_DIR,
                "random_vs_grouped_F1.png"
            ),
            dpi=200
        )


        plt.close()


else:

    print()
    print(
        "WARNING:"
    )

    print(
        "Random experiment2_results.csv "
        "was not found."
    )

    print(
        "Grouped results were still saved."
    )


# ============================================================
# 27. Save Configuration
# ============================================================

config = {

    "random_state":
        RANDOM_STATE,

    "split_ratio": {
        "train": 0.75,
        "validation": 0.15,
        "test": 0.10
    },

    "normalization": {
        "lowercase": True,
        "strip_whitespace": True,
        "remove_fragment": True,
        "add_http_scheme": True,
        "group_definition":
            "exact match after normalization"
    },

    "total_rows":
        int(num_rows),

    "unique_groups":
        int(num_unique_groups),

    "duplicate_rows":
        int(num_duplicate_rows),

    "duplicate_rate":
        float(duplicate_rate),

    "train_size":
        int(len(train_idx)),

    "validation_size":
        int(len(val_idx)),

    "test_size":
        int(len(test_idx)),

    "models": [
        "CNN grouped adaptation",
        "MLP grouped adaptation",
        "Intermediate fusion grouped adaptation"
    ],

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "early_stopping_patience":
        PATIENCE

}


with open(
    os.path.join(
        GROUPED_DIR,
        "grouped_sensitivity_config.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 28. Final Summary
# ============================================================

print()
print("=" * 70)
print("GROUPED SENSITIVITY ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    "Results:",
    results_path
)

print(
    "Output directory:",
    GROUPED_DIR
)

print()

print(
    "Saved files:"
)

print(
    " - normalized_url_groups.csv"
)

print(
    " - grouped_split_assignment.csv"
)

print(
    " - grouped_train_idx.npy"
)

print(
    " - grouped_val_idx.npy"
)

print(
    " - grouped_test_idx.npy"
)

print(
    " - grouped_sensitivity_results.csv"
)

print(
    " - grouped_test_predictions.csv"
)

print(
    " - random_vs_grouped_comparison.csv"
)

print(
    " - random_vs_grouped_F1.png"
)

print(
    " - grouped_sensitivity_config.json"
)

print()

print(
    "No normalized URL group leakage detected."
)

print(
    "Done."
)

