# ============================================================
# 1. Import Libraries
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ============================================================
# 2. Google Drive Path
# ============================================================

# Dataset 路徑
CSV_PATH = '/content/drive/MyDrive/phishing_project/cnn/malicious_phish.csv'

# CNN 專案資料夾
BASE_DIR = '/content/drive/MyDrive/phishing_project/cnn'

# 如果資料夾不存在，就建立
os.makedirs(BASE_DIR, exist_ok=True)

print("CNN 專案資料夾：")
print(BASE_DIR)


# ============================================================
# 3. Load Dataset
# ============================================================

print("\n==============================")
print("Loading Dataset")
print("==============================")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. Basic Dataset Check
# ============================================================

print("\n==============================")
print("Dataset Information")
print("==============================")

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df['type'].value_counts())


# ============================================================
# 5. Extract URL and Label
# ============================================================

urls = df['url'].astype(str)
labels = df['type'].astype(str)


# ============================================================
# 6. Encode Labels
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

print("\n==============================")
print("Label Encoding")
print("==============================")

print("Classes:")
print(label_encoder.classes_)

print("Number of classes:", num_classes)

print("\nEncoded label distribution:")
print(pd.Series(y).value_counts().sort_index())


# ============================================================
# 7. Train / Validation / Test Split
#
# 75% Training
# 15% Validation
# 10% Test
# ============================================================

RANDOM_STATE = 42
# 第一階段：
# 75% train
# 25% temporary
X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    urls,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

# 第二階段：
# temporary 的 60% = 15% validation
# temporary 的 40% = 10% test
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text,
    y_temp,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


# ============================================================
# 8. Check Split Size
# ============================================================

print("\n==============================")
print("Dataset Split")
print("==============================")

print("Training samples:", len(X_train_text))
print("Validation samples:", len(X_val_text))
print("Test samples:", len(X_test_text))

total = len(urls)

print("\nActual percentages:")
print("Train:", len(X_train_text) / total)
print("Validation:", len(X_val_text) / total)
print("Test:", len(X_test_text) / total)


# ============================================================
# 9. Check Class Distribution
# ============================================================

print("\n==============================")
print("Class Distribution")
print("==============================")

print("\nTraining:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nValidation:")
print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
)

print("\nTest:")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. Character-level Tokenizer
# ============================================================

# ------------------------------------------------------------
# 原本如果在 split 前：
#
# tokenizer.fit_on_texts(urls)
#
# 會讓 validation / test 的字元資訊提前進入 tokenizer。
#
# 因此這裡我只使用 Training Data 建立 vocabulary。
# ------------------------------------------------------------

tokenizer = Tokenizer(
    char_level=True,
    lower=True
)

tokenizer.fit_on_texts(X_train_text)


# ============================================================
# 11. Vocabulary Information
# ============================================================

vocab_size = len(tokenizer.word_index) + 1

print("\n==============================")
print("Tokenizer")
print("==============================")

print("Vocabulary size:", vocab_size)

print("\nCharacter mapping:")
print(dict(list(tokenizer.word_index.items())[:30]))


# ============================================================
# 12. Convert URLs to Sequences
# ============================================================

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_val_seq = tokenizer.texts_to_sequences(X_val_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

'''
# ============================================================
# 13. URL Length Analysis
# ============================================================

train_url_lengths = X_train_text.apply(len)

print("\n==============================")
print("URL Length Analysis")
print("==============================")

print("Mean:",
      train_url_lengths.mean())

print("Median:",
      train_url_lengths.median())

print("95th percentile:",
      train_url_lengths.quantile(0.95))

print("97th percentile:",
      train_url_lengths.quantile(0.97))

print("99th percentile:",
      train_url_lengths.quantile(0.99))

print("Maximum:",
      train_url_lengths.max())


# ============================================================
# 14. Candidate Maximum Lengths
# ============================================================

candidate_lengths = [100, 120, 150, 160, 200, 250]

print("\n==============================")
print("Candidate Max Lengths")
print("==============================")

for length in candidate_lengths:

    truncated_ratio = (
        (train_url_lengths > length).mean()
    )

    print(
        f"max_length={length}: "
        f"truncation={truncated_ratio:.4%}"
    )
'''

# ============================================================
# 15. Select Maximum Sequence Length
# ============================================================

# 13.14.的部分我已自行跑過，length設定150是可行的。
MAX_LENGTH = 150

print("\nSelected MAX_LENGTH:", MAX_LENGTH)


# ============================================================
# 16. Padding / Truncation
# ============================================================

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

X_val_pad = pad_sequences(
    X_val_seq,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)


# ============================================================
# 17. Check Input Shape
# ============================================================

print("\n==============================")
print("Input Shape")
print("==============================")

print("X_train:", X_train_pad.shape)
print("X_val:", X_val_pad.shape)
print("X_test:", X_test_pad.shape)


# ============================================================
# 18. Calculate Padding Ratio
# ============================================================

average_padding = np.mean(
    np.sum(X_train_pad == 0, axis=1)
)

padding_ratio = (
    average_padding / MAX_LENGTH
)

print("\nAverage padding length:",
      average_padding)

print("Average padding ratio:",
      f"{padding_ratio:.2%}")


# ============================================================
# 19. Build 1D CNN Model
# ============================================================

print("\n==============================")
print("Building CNN Model")
print("==============================")


model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=32
    ),

    # Extract local character patterns
    Conv1D(
        filters=128,
        kernel_size=5,
        activation='relu'
    ),

    # Extract strongest activation
    GlobalMaxPooling1D(),

    # Fully connected layer
    Dense(
        64,
        activation='relu'
    ),

    # Reduce overfitting
    Dropout(
        0.5
    ),

    # Output layer
    Dense(
        num_classes,
        activation='softmax'
    )
])


# ============================================================
# 20. Compile Model
# ============================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 21. Model Summary
# ============================================================

print("\n==============================")
print("Model Summary")
print("==============================")

model.build(
    input_shape=(None, MAX_LENGTH)
)

model.summary()


# ============================================================
# 22. Callbacks
# ============================================================

# ------------------------------------------------------------
# 這裡我設定如果 validation loss 連續3個 epoch 沒有改善，就停止 training。


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# ------------------------------------------------------------
# ModelCheckpoint
#
# 只保存 validation loss 最好的模型。
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    'malicious_url_cnn.keras'
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)


print("\nModel will be saved to:")
print(MODEL_PATH)


# ============================================================
# 23. Training Configuration
# ============================================================

EPOCHS = 20
BATCH_SIZE = 128


# ============================================================
# 24. Train Model
# ============================================================

print("\n==============================")
print("Training CNN")
print("==============================")


history = model.fit(
    X_train_pad,
    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        X_val_pad,
        y_val
    ),

    callbacks=[
        early_stop,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# 25. Training History
# ============================================================

print("\n==============================")
print("Training History")
print("==============================")

print("Epochs actually trained:",
      len(history.history['loss']))

print("\nBest validation loss:")

best_epoch_loss = (
    np.argmin(history.history['val_loss']) + 1
)

best_val_loss = min(
    history.history['val_loss']
)

print("Epoch:", best_epoch_loss)
print("Val Loss:", best_val_loss)


best_epoch_acc = (
    np.argmax(history.history['val_accuracy']) + 1
)

best_val_acc = max(
    history.history['val_accuracy']
)

print("\nBest validation accuracy:")
print("Epoch:", best_epoch_acc)
print("Val Accuracy:", best_val_acc)


# ============================================================
# 26. Save Training History
# ============================================================

training_history = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'best_val_loss_epoch': best_epoch_loss,
    'best_val_loss': best_val_loss,
    'best_val_accuracy_epoch': best_epoch_acc,
    'best_val_accuracy': best_val_acc
}

HISTORY_PATH = os.path.join(
    BASE_DIR,
    'training_history.pkl'
)

with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(
        training_history,
        f
    )

print("\nTraining history saved to:")
print(HISTORY_PATH)


# ============================================================
# 27. Evaluate Validation Set
# ============================================================

print("\n==============================")
print("Validation Evaluation")
print("==============================")


val_probabilities = model.predict(
    X_val_pad,
    batch_size=BATCH_SIZE,
    verbose=1
)

val_predictions = np.argmax(
    val_probabilities,
    axis=1
)


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average='macro'
)

val_weighted_f1 = f1_score(
    y_val,
    val_predictions,
    average='weighted'
)


print("\nValidation Accuracy:",
      val_accuracy)

print("Validation Macro F1:",
      val_macro_f1)

print("Validation Weighted F1:",
      val_weighted_f1)


print("\nValidation Classification Report:")

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=label_encoder.classes_
    )
)


print("\nValidation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        val_predictions
    )
)


# ============================================================
# 28. Test Set Prediction
# ============================================================

print("\n==============================")
print("Test Evaluation")
print("==============================")


test_probabilities = model.predict(
    X_test_pad,
    batch_size=BATCH_SIZE,
    verbose=1
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)


# ============================================================
# 29. Test Metrics
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average='macro'
)

test_weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average='weighted'
)


print("\n==============================")
print("Test Results")
print("==============================")

print("Test Accuracy:",
      test_accuracy)

print("Test Macro F1:",
      test_macro_f1)

print("Test Weighted F1:",
      test_weighted_f1)


# ============================================================
# 30. Classification Report
# ============================================================

print("\n==============================")
print("Classification Report")
print("==============================")


print(
    classification_report(
        y_test,
        test_predictions,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 31. Confusion Matrix
# ============================================================

print("\n==============================")
print("Confusion Matrix")
print("==============================")


cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)


# ============================================================
# 32. Save Tokenizer
# ============================================================

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    'tokenizer.pkl'
)

with open(
    TOKENIZER_PATH,
    'wb'
) as f:

    pickle.dump(
        tokenizer,
        f
    )


print("\nTokenizer saved to:")
print(TOKENIZER_PATH)


# ============================================================
# 33. Save Label Encoder
# ============================================================

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    'label_encoder.pkl'
)

with open(
    LABEL_ENCODER_PATH,
    'wb'
) as f:

    pickle.dump(
        label_encoder,
        f
    )


print("\nLabel encoder saved to:")
print(LABEL_ENCODER_PATH)


# ============================================================
# 34. Save Training Configuration
# ============================================================

training_config = {

    # Dataset
    'dataset': 'malicious_phish.csv',

    # Data split
    'train_ratio': 0.75,
    'validation_ratio': 0.15,
    'test_ratio': 0.10,

    # Reproducibility
    'random_state': RANDOM_STATE,

    # Tokenizer
    'tokenizer_type': 'character-level',
    'char_level': True,
    'lower': True,

    # Sequence
    'max_length': MAX_LENGTH,
    'padding': 'post',
    'truncating': 'post',

    # Model
    'embedding_output_dim': 32,
    'conv_filters': 128,
    'conv_kernel_size': 5,
    'dense_units': 64,
    'dropout': 0.5,

    # Training
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,

    # Early stopping
    'early_stopping_monitor': 'val_loss',
    'early_stopping_patience': 3,
    'restore_best_weights': True,

    # Model checkpoint
    'checkpoint_monitor': 'val_loss',
    'checkpoint_mode': 'min',

    # Dataset information
    'vocab_size': vocab_size,
    'num_classes': num_classes,
    'classes': label_encoder.classes_.tolist(),

    # Results
    'validation_accuracy': float(val_accuracy),
    'validation_macro_f1': float(val_macro_f1),
    'validation_weighted_f1': float(val_weighted_f1),

    'test_accuracy': float(test_accuracy),
    'test_macro_f1': float(test_macro_f1),
    'test_weighted_f1': float(test_weighted_f1),

    # Best epochs
    'best_val_loss_epoch': int(best_epoch_loss),
    'best_val_loss': float(best_val_loss),

    'best_val_accuracy_epoch': int(best_epoch_acc),
    'best_val_accuracy': float(best_val_acc)
}


CONFIG_PATH = os.path.join(
    BASE_DIR,
    'training_config.pkl'
)

with open(
    CONFIG_PATH,
    'wb'
) as f:

    pickle.dump(
        training_config,
        f
    )


print("\nTraining configuration saved to:")
print(CONFIG_PATH)


# ============================================================
# 35. Save Test Predictions
# ============================================================

# Convert encoded labels back to original labels

true_labels = label_encoder.inverse_transform(
    y_test
)

predicted_labels = label_encoder.inverse_transform(
    test_predictions
)


prediction_df = pd.DataFrame({

    'url': X_test_text.values,

    'true_label': true_labels,

    'predicted_label': predicted_labels,

    'correct': (
        true_labels == predicted_labels
    )
})


# ------------------------------------------------------------
# Optional:
# Save prediction probability for every class
# ------------------------------------------------------------

for i, class_name in enumerate(
    label_encoder.classes_
):

    prediction_df[
        f'prob_{class_name}'
    ] = test_probabilities[:, i]


PREDICTION_PATH = os.path.join(
    BASE_DIR,
    'test_predictions.csv'
)

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


print("\nTest predictions saved to:")
print(PREDICTION_PATH)


# ============================================================
# 36. Save Final Model Explicitly
# ============================================================
if os.path.exists(MODEL_PATH):

    print("\nBest model successfully saved.")

else:

    print("\nWARNING: Model file was not found.")


# ============================================================
# 37. Final Summary
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nDataset:")
print(CSV_PATH)

print("\nSplit:")
print("Train:      75%")
print("Validation: 15%")
print("Test:       10%")

print("\nModel:")
print("Character-level 1D CNN")

print("\nVocabulary size:")
print(vocab_size)

print("\nMaximum sequence length:")
print(MAX_LENGTH)

print("\nNumber of classes:")
print(num_classes)

print("\nClasses:")
print(label_encoder.classes_)

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")

print("\nTest Macro F1:")
print(f"{test_macro_f1:.4f}")

print("\nTest Weighted F1:")
print(f"{test_weighted_f1:.4f}")


print("\n")
print("=" * 60)
print("SAVED FILES")
print("=" * 60)

print("\n1. Model:")
print(MODEL_PATH)

print("\n2. Tokenizer:")
print(TOKENIZER_PATH)

print("\n3. Label Encoder:")
print(LABEL_ENCODER_PATH)

print("\n4. Training History:")
print(HISTORY_PATH)

print("\n5. Training Config:")
print(CONFIG_PATH)

print("\n6. Test Predictions:")
print(PREDICTION_PATH)


print("\n")
print("=" * 60)
print("CNN TRAINING COMPLETE")
print("=" * 60)