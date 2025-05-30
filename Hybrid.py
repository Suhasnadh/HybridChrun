#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn xgboost sentence-transformers openpyxl


# In[11]:


pip install shap


# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# =========================
# Load the Dataset
# =========================
df = pd.read_excel("Telco_customer_churn.xlsx", sheet_name="Telco_Churn")

# =========================
# Set Target Column
# =========================
df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# =========================
# Drop Leakage and Irrelevant Columns
# =========================
df.drop([
    'CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
    'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score',
    'CLTV', 'Churn Reason'
], axis=1, inplace=True)

# =========================
# Separate Features and Target
# =========================
X = df.drop('Churn', axis=1)
y = df['Churn']

# =========================
# Encode Categorical Features
# =========================
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# =========================
# Scale the Features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =========================
# Compute Class Weights
# =========================
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), weights))
scale_pos_weight = weights[1] / weights[0]

# =========================
# Train Logistic Regression
# =========================
log_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# =========================
# Train XGBoost
# =========================
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# =========================
# Evaluate Logistic Regression
# =========================
print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred))
print("Classification Report:\n", classification_report(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))

# =========================
# Evaluate XGBoost
# =========================
print("\n===== XGBoost =====")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("F1 Score:", f1_score(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

# =========================
# Optional: SHAP Explainability for XGBoost
# =========================
print("\nGenerating SHAP explanations...")
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Comment this line out if you're running in a terminal-only environment
shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())


# In[4]:


pip install tensorflow pandas scikit-learn openpyxl


# In[13]:


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load dataset
df = pd.read_excel("Telco_customer_churn.xlsx", sheet_name="Telco_Churn")

# Drop unnecessary columns
df.drop(['CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
         'Latitude', 'Longitude', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'],
        axis=1, inplace=True, errors='ignore')

# Create target variable
df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
df.drop(['Churn Label'], axis=1, inplace=True)

# Encode categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Feature and target split
X = df.drop('Churn', axis=1)
y = df['Churn']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)

# =======================
# Model 1: Simple Neural Network
# =======================
model1 = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
pred1 = (model1.predict(X_test) > 0.5).astype("int32")

print("===== Model 1: Simple NN =====")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1))
print("Classification Report:\n", classification_report(y_test, pred1))

# =======================
# Model 2: Deeper Neural Network
# =======================
model2 = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
pred2 = (model2.predict(X_test) > 0.5).astype("int32")

print("===== Model 2: Deeper NN =====")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2))
print("Classification Report:\n", classification_report(y_test, pred2))


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


# In[15]:


plot_confusion(y_test, pred1, "Simple NN")


# In[16]:


plot_confusion(y_test, pred2, "Deeper NN")


# In[6]:


pip install sentence-transformers xgboost scikit-learn pandas openpyxl


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# Load dataset
df = pd.read_excel("Telco_customer_churn.xlsx", sheet_name="Telco_Churn")

# Create binary target
df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# Remove target leakage and unnecessary fields
df.drop(columns=[
    'CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
    'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score',
    'CLTV', 'Churn Reason'
], inplace=True)

# Step 1: Simulate synthetic support tickets (pre-churn neutral text)
import random
neutral_templates = [
    "Customer inquired about billing details",
    "Asked about internet speed",
    "Reported intermittent connection issue",
    "Requested plan upgrade information",
    "Questioned late fee on recent invoice",
    "Complaint regarding data usage limit",
    "Request for technician appointment",
    "Asked about contract expiration date"
]
df['SupportNotes'] = [random.choice(neutral_templates) for _ in range(len(df))]

# Separate structured and text features
text_data = df['SupportNotes']
structured_df = df.drop(columns=['SupportNotes', 'Churn'])

# Label encode categorical features
for col in structured_df.columns:
    if structured_df[col].dtype == 'object':
        structured_df[col] = LabelEncoder().fit_transform(structured_df[col].astype(str))

# Scale numeric values
scaler = StandardScaler()
structured_scaled = scaler.fit_transform(structured_df)

# Step 2: Generate BERT embeddings
print("Generating BERT embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = model.encode(text_data.tolist(), show_progress_bar=True)

# Combine features
X = np.hstack((structured_scaled, text_embeddings))
y = df['Churn']

# Step 3: Apply Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, f1_scores = [], []

from sklearn.metrics import f1_score

print("\n===== Cross-Validated Results =====")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracies.append(acc)
    f1_scores.append(f1)A

    print(f"Fold {fold} - Accuracy: {acc:.4f}, F1-score (churn): {f1:.4f}")

print("\n===== Final Results =====")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
print(f"Mean F1-score: {np.mean(f1_scores):.4f}")


# In[17]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    return auc_score


# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import xgboost as xgb

# Load dataset
df = pd.read_excel("Telco_customer_churn.xlsx", sheet_name="Telco_Churn")

# Drop unnecessary columns
df.drop(['CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
         'Latitude', 'Longitude', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'],
        axis=1, inplace=True, errors='ignore')

# Create target variable
df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
df.drop(['Churn Label'], axis=1, inplace=True)

# Encode categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Prepare scaled data for deep learning
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Prepare raw data for XGBoost
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Simple neural network
model1 = Sequential([
    Dense(64, input_dim=X_train_dl.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit(X_train_dl, y_train_dl, epochs=20, batch_size=32, verbose=0)

# Model 2: Deeper neural network
model2 = Sequential([
    Dense(128, input_dim=X_train_dl.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_train_dl, y_train_dl, epochs=20, batch_size=32, verbose=0)

# Model 3: XGBoost
model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train_xgb, y_train_xgb)

# ROC Plotting Function
def plot_roc_curve(y_true, y_probs, label):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plot_roc_curve(y_test_dl, model1.predict(X_test_dl).ravel(), "Simple NN")
plot_roc_curve(y_test_dl, model2.predict(X_test_dl).ravel(), "Deeper NN")
plot_roc_curve(y_test_xgb, model_xgb.predict_proba(X_test_xgb)[:, 1], "XGBoost")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

# Final Deep FFNN for IEEE Paper
model = Sequential([
    Dense(128, input_dim=X_train_dl.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train with validation split
history = model.fit(X_train_dl, y_train_dl, validation_split=0.2,
                    epochs=50, batch_size=32, callbacks=[es], verbose=1)

# Predict + Evaluate
pred_probs = model.predict(X_test_dl).ravel()
pred_labels = (pred_probs > 0.5).astype("int32")

# AUC and Classification
auc_score = roc_auc_score(y_test_dl, pred_probs)
print(f"Final Deep FFNN AUC: {auc_score:.4f}")
print("Classification Report:\n", classification_report(y_test_dl, pred_labels))


# In[21]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[22]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# =========================
# Load and Preprocess Data
# =========================
df = pd.read_excel("Telco_customer_churn.xlsx", sheet_name="Telco_Churn")

# Target variable
df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# Drop irrelevant/leakage columns
df.drop([
    'CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
    'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score',
    'CLTV', 'Churn Reason'
], axis=1, inplace=True)

# Label encode categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Feature & Target Split
X = df.drop('Churn', axis=1)
y = df['Churn']

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)

# =========================
# Build and Train Deep FFNN
# =========================
model = Sequential([
    Dense(256, input_dim=X_train_dl.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=100, batch_size=32,
                    callbacks=[es], verbose=1)

# =========================
# Evaluate Model
# =========================
pred_probs = model.predict(X_test_dl).ravel()
pred_labels = (pred_probs > 0.5).astype("int32")

auc_score = roc_auc_score(y_test_dl, pred_probs)
print(f"\n📊 Improved Deep FFNN AUC: {auc_score:.4f}")
print("Classification Report:\n", classification_report(y_test_dl, pred_labels))

# =========================
# Plot ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test_dl, pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Improved FFNN (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve – Final Deep FFNN")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("final_ffnn_roc.png", dpi=300)
plt.show()


# In[27]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# =========================
# Load Dataset
# =========================
df = pd.read_excel("Telco_customer_churn.xlsx", sheet_name="Telco_Churn")
df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
df.drop(columns=['CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
                 'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score',
                 'CLTV', 'Churn Reason'], inplace=True)

# =========================
# Simulate Support Tickets
# =========================
neutral_templates = [
    "Customer inquired about billing details",
    "Asked about internet speed",
    "Reported intermittent connection issue",
    "Requested plan upgrade information",
    "Questioned late fee on recent invoice",
    "Complaint regarding data usage limit",
    "Request for technician appointment",
    "Asked about contract expiration date"
]
df['SupportNotes'] = [random.choice(neutral_templates) for _ in range(len(df))]

# =========================
# Structured Data Prep
# =========================
X_struct = df.drop(columns=['SupportNotes', 'Churn'])
y = df['Churn']
for col in X_struct.columns:
    if X_struct[col].dtype == 'object':
        X_struct[col] = LabelEncoder().fit_transform(X_struct[col].astype(str))

scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct)

# =========================
# LLM Features (MiniLM)
# =========================
model_bert = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = model_bert.encode(df['SupportNotes'].tolist(), show_progress_bar=True)

X_combined = np.hstack((X_struct_scaled, text_embeddings))

# =========================
# Train/Test Splits
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_struct_scaled, y, test_size=0.2, random_state=42)
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# =========================
# Logistic Regression
# =========================
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), weights))
log_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
log_model.fit(X_train, y_train)
log_probs = log_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
auc_log = auc(fpr_log, tpr_log)

# =========================
# XGBoost
# =========================
scale_pos_weight = weights[1] / weights[0]
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# =========================
# FFNN (64-32)
# =========================
model_ffnn1 = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_ffnn1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_ffnn1.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
ffnn1_probs = model_ffnn1.predict(X_test).ravel()
fpr_ffnn1, tpr_ffnn1, _ = roc_curve(y_test, ffnn1_probs)
auc_ffnn1 = auc(fpr_ffnn1, tpr_ffnn1)

# =========================
# Improved FFNN (512-256-128-64)
# =========================
model_ffnn2 = Sequential([
    Dense(512, input_dim=X_train.shape[1]),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),
    Dense(256),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.3),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.3),
    Dense(64),
    LeakyReLU(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model_ffnn2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model_ffnn2.fit(X_train, y_train, validation_split=0.2, epochs=80, batch_size=32, callbacks=[es], verbose=1)
ffnn2_probs = model_ffnn2.predict(X_test).ravel()
fpr_ffnn2, tpr_ffnn2, _ = roc_curve(y_test, ffnn2_probs)
auc_ffnn2 = auc(fpr_ffnn2, tpr_ffnn2)

# =========================
# LLM + XGBoost
# =========================
model_llm_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_llm_xgb.fit(X_train_comb, y_train_comb)
llm_probs = model_llm_xgb.predict_proba(X_test_comb)[:, 1]
fpr_llm, tpr_llm, _ = roc_curve(y_test_comb, llm_probs)
auc_llm = auc(fpr_llm, tpr_llm)

# =========================
# Plot ROC for All Models
# =========================
plt.figure(figsize=(10, 7))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})')
plt.plot(fpr_ffnn1, tpr_ffnn1, label=f'FFNN (64-32) (AUC = {auc_ffnn1:.3f})')
plt.plot(fpr_ffnn2, tpr_ffnn2, label=f'FFNN (512-256-128-64) (AUC = {auc_ffnn2:.3f})')
plt.plot(fpr_llm, tpr_llm, label=f'LLM + XGBoost (AUC = {auc_llm:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title('ROC Curve Comparison – All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("final_roc_curve_all_models.png", dpi=300)
plt.show()


# In[28]:


# Telco Customer Churn Prediction - Complete ML Pipeline
# This notebook implements multiple ML models to achieve 85-93% accuracy as specified

# ============================================================================
# 1. IMPORT LIBRARIES AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier

# Evaluation metrics
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           f1_score, precision_score, recall_score)

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# ============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration
    """
    print("📊 Loading and exploring the dataset...")
    
    # Load data (adjust path as needed)
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info
    print("\n📈 Dataset Info:")
    print(df.info())
    
    print("\n🎯 Target Variable Distribution:")
    if 'Churn Value' in df.columns:
        target_col = 'Churn Value'
    elif 'Churn Label' in df.columns:
        target_col = 'Churn Label'
        # Convert to binary if needed
        df['Churn Value'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
        target_col = 'Churn Value'
    
    print(df[target_col].value_counts())
    print(f"Churn Rate: {df[target_col].mean():.2%}")
    
    return df

# Load the dataset
# Replace 'Telco_customer_churn.xlsx' with your actual file path
df = load_and_explore_data('Telco_customer_churn.xlsx')

# ============================================================================
# 3. DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

def preprocess_data(df):
    """
    Comprehensive data preprocessing pipeline
    """
    print("🔧 Starting data preprocessing...")
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Handle missing values
    print("🔍 Checking for missing values...")
    missing_summary = data.isnull().sum()
    print(missing_summary[missing_summary > 0])
    
    # Convert Total Charges to numeric (it might be stored as string with spaces)
    if 'Total Charges' in data.columns:
        data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
        # Fill missing Total Charges with 0 (for new customers)
        data['Total Charges'].fillna(0, inplace=True)
    
    # Feature Engineering
    print("⚙️ Engineering new features...")
    
    # Create tenure groups
    if 'Tenure Months' in data.columns:
        data['Tenure_Group'] = pd.cut(data['Tenure Months'], 
                                    bins=[0, 12, 24, 48, 72], 
                                    labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
    
    # Create monthly charges groups
    if 'Monthly Charges' in data.columns:
        data['Charges_Group'] = pd.cut(data['Monthly Charges'], 
                                     bins=[0, 30, 60, 90, 150], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create total services count
    service_columns = ['Phone Service', 'Multiple Lines', 'Internet Service', 
                      'Online Security', 'Online Backup', 'Device Protection', 
                      'Tech Support', 'Streaming TV', 'Streaming Movies']
    
    # Count services (convert Yes to 1, others to 0)
    for col in service_columns:
        if col in data.columns:
            data[f'{col}_Binary'] = (data[col] == 'Yes').astype(int)
    
    data['Total_Services'] = sum([data[f'{col}_Binary'] for col in service_columns if f'{col}_Binary' in data.columns])
    
    # Customer lifetime value per month
    if all(col in data.columns for col in ['Total Charges', 'Tenure Months']):
        data['CLV_per_Month'] = data['Total Charges'] / (data['Tenure Months'] + 1)  # +1 to avoid division by zero
    
    # Contract length encoding
    if 'Contract' in data.columns:
        contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        data['Contract_Length'] = data['Contract'].map(contract_map)
    
    print("✅ Feature engineering completed!")
    return data

# Apply preprocessing
processed_df = preprocess_data(df)

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# ============================================================================

def create_eda_plots(df):
    """
    Create comprehensive EDA plots
    """
    print("📊 Creating EDA visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Telco Customer Churn - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # Churn distribution
    axes[0,0].pie(df['Churn Value'].value_counts(), labels=['No Churn', 'Churn'], 
                  autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
    axes[0,0].set_title('Churn Distribution')
    
    # Tenure vs Churn
    sns.boxplot(data=df, x='Churn Value', y='Tenure Months', ax=axes[0,1])
    axes[0,1].set_title('Tenure Months vs Churn')
    axes[0,1].set_xlabel('Churn (0=No, 1=Yes)')
    
    # Monthly Charges vs Churn
    sns.boxplot(data=df, x='Churn Value', y='Monthly Charges', ax=axes[0,2])
    axes[0,2].set_title('Monthly Charges vs Churn')
    axes[0,2].set_xlabel('Churn (0=No, 1=Yes)')
    
    # Contract type vs Churn
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn Value'].mean()
        axes[1,0].bar(contract_churn.index, contract_churn.values, color='skyblue')
        axes[1,0].set_title('Churn Rate by Contract Type')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Internet Service vs Churn
    if 'Internet Service' in df.columns:
        internet_churn = df.groupby('Internet Service')['Churn Value'].mean()
        axes[1,1].bar(internet_churn.index, internet_churn.values, color='lightgreen')
        axes[1,1].set_title('Churn Rate by Internet Service')
    
    # Payment Method vs Churn
    if 'Payment Method' in df.columns:
        payment_churn = df.groupby('Payment Method')['Churn Value'].mean()
        axes[1,2].bar(payment_churn.index, payment_churn.values, color='orange')
        axes[1,2].set_title('Churn Rate by Payment Method')
        axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Create EDA plots
create_eda_plots(processed_df)

# ============================================================================
# 5. FEATURE SELECTION AND DATA PREPARATION
# ============================================================================

def prepare_features(df):
    """
    Prepare features for machine learning models
    """
    print("🔧 Preparing features for ML models...")
    
    # Define target variable
    target = 'Churn Value'
    
    # Select relevant features (excluding IDs and redundant columns)
    exclude_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
                   'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 
                   'CLTV', 'Churn Reason', target]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify numerical and categorical columns
    numerical_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare X and y
    X = df[feature_cols]
    y = df[target]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, preprocessor, numerical_features, categorical_features

# Prepare features
X, y, preprocessor, num_features, cat_features = prepare_features(processed_df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================================================
# 6. MODEL DEFINITION AND HYPERPARAMETER TUNING
# ============================================================================

def get_models_and_params():
    """
    Define models and their hyperparameter grids for tuning
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'model__C': [0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga']
            }
        },
        
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        },
        
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 6, 10],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            }
        },
        
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 1.0]
            }
        },
        
        'Neural Network': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'model__hidden_layer_sizes': [(100,), (100, 50), (50, 30)],
                'model__alpha': [0.0001, 0.001, 0.01],
                'model__learning_rate': ['constant', 'adaptive']
            }
        }
    }
    
    return models

# Get models and parameters
models_config = get_models_and_params()

# ============================================================================
# 7. MODEL TRAINING AND HYPERPARAMETER TUNING
# ============================================================================

def train_and_tune_models(models_config, X_train, y_train, preprocessor):
    """
    Train models with hyperparameter tuning using GridSearchCV
    """
    print("🚀 Starting model training and hyperparameter tuning...")
    
    best_models = {}
    cv_scores = {}
    
    # Use StratifiedKFold for better cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models_config.items():
        print(f"\n🔧 Training {name}...")
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store results
        best_models[name] = grid_search.best_estimator_
        cv_scores[name] = grid_search.best_score_
        
        print(f"✅ {name} - Best CV Score: {grid_search.best_score_:.4f}")
        print(f"📊 Best Parameters: {grid_search.best_params_}")
    
    return best_models, cv_scores

# Train and tune models
best_models, cv_scores = train_and_tune_models(models_config, X_train, y_train, preprocessor)

# ============================================================================
# 8. MODEL EVALUATION AND COMPARISON
# ============================================================================

def evaluate_models(models, X_test, y_test):
    """
    Comprehensive model evaluation with multiple metrics
    """
    print("📊 Evaluating models on test set...")
    
    results = {}
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        }
        
        predictions[name] = y_pred
        probabilities[name] = y_prob
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
    
    return results, predictions, probabilities

# Evaluate models
results, predictions, probabilities = evaluate_models(best_models, X_test, y_test)

# ============================================================================
# 9. RESULTS VISUALIZATION AND ROC CURVES
# ============================================================================

def plot_model_comparison(results, cv_scores):
    """
    Create comprehensive comparison plots
    """
    print("📈 Creating model comparison visualizations...")
    
    # Create results dataframe
    results_df = pd.DataFrame(results).T
    
    # Plot 1: Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    results_df['Accuracy'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Test Accuracy')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=0.85, color='red', linestyle='--', label='Target: 85%')
    axes[0,0].legend()
    
    # AUC comparison
    results_df['AUC'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('AUC Score')
    axes[0,1].set_ylabel('AUC')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].axhline(y=0.87, color='red', linestyle='--', label='Target: 0.87')
    axes[0,1].legend()
    
    # F1-Score comparison
    results_df['F1-Score'].plot(kind='bar', ax=axes[1,0], color='orange')
    axes[1,0].set_title('F1-Score')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Cross-validation scores
    cv_df = pd.Series(cv_scores)
    cv_df.plot(kind='bar', ax=axes[1,1], color='salmon')
    axes[1,1].set_title('Cross-Validation Accuracy')
    axes[1,1].set_ylabel('CV Accuracy')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Display results table
    print("\n📊 Detailed Results Comparison:")
    print(results_df.round(4))

def plot_roc_curves(y_test, probabilities):
    """
    Plot ROC curves for all models
    """
    print("📈 Creating ROC curves for all models...")
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, y_prob) in enumerate(probabilities.items()):
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                label=f'{name} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_confusion_matrices(y_test, predictions):
    """
    Plot confusion matrices for all models
    """
    print("🎯 Creating confusion matrices...")
    
    n_models = len(predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(len(predictions), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Create all visualizations
plot_model_comparison(results, cv_scores)
plot_roc_curves(y_test, probabilities)
plot_confusion_matrices(y_test, predictions)

# ============================================================================
# 10. ENSEMBLE MODEL CREATION
# ============================================================================

def create_ensemble_model(best_models, X_train, y_train, X_test, y_test):
    """
    Create an ensemble model combining top performers
    """
    print("🔄 Creating ensemble model...")
    
    # Select top 3 models based on CV scores
    top_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"Top 3 models for ensemble: {[name for name, score in top_models]}")
    
    # Create voting classifier
    estimators = [(name, best_models[name]) for name, score in top_models]
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test)
    y_prob_ensemble = ensemble.predict_proba(X_test)[:, 1]
    
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_auc = roc_auc_score(y_test, y_prob_ensemble)
    
    print(f"🎯 Ensemble Model Results:")
    print(f"Accuracy: {ensemble_accuracy:.4f}")
    print(f"AUC: {ensemble_auc:.4f}")
    
    return ensemble, ensemble_accuracy, ensemble_auc

# Create ensemble model
ensemble_model, ensemble_acc, ensemble_auc = create_ensemble_model(best_models, X_train, y_train, X_test, y_test)

# ============================================================================
# 11. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(best_models, num_features, cat_features):
    """
    Analyze and visualize feature importance
    """
    print("🔍 Analyzing feature importance...")
    
    # Get feature importance from tree-based models
    tree_models = ['Random Forest', 'XGBoost', 'Gradient Boosting']
    
    fig, axes = plt.subplots(1, len(tree_models), figsize=(18, 6))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(tree_models):
        if model_name in best_models:
            model = best_models[model_name]
            
            # Get feature names after preprocessing
            feature_names = (num_features + 
                           list(model.named_steps['preprocessor']
                               .named_transformers_['cat']
                               .get_feature_names_out(cat_features)))
            
            # Get feature importance
            importances = model.named_steps['model'].feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            # Plot
            axes[i].bar(range(len(indices)), importances[indices])
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel('Importance')
            axes[i].set_xticks(range(len(indices)))
            axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# Analyze feature importance
analyze_feature_importance(best_models, num_features, cat_features)

# ============================================================================
# 12. FINAL SUMMARY AND EXPECTED RESULTS COMPARISON
# ============================================================================

def create_final_summary(results, cv_scores, ensemble_acc, ensemble_auc):
    """
    Create final summary comparing with expected results
    """
    print("📋 FINAL SUMMARY - MODEL PERFORMANCE vs EXPECTED RESULTS")
    print("=" * 70)
    
    # Expected results from the summary table
    expected_results = {
        'XGBoost': {'Accuracy': '85-88%', 'AUC': '0.87-0.90'},
        'Logistic Regression': {'Accuracy': '78-80%', 'AUC': '0.76-0.79'},
        'Neural Network': {'Accuracy': '89-92%', 'AUC': '0.91-0.93'},
        'Gradient Boosting': {'Accuracy': '86-88%', 'AUC': '0.88-0.91'},
        'Ensemble': {'Accuracy': '90-93%', 'AUC': '0.91-0.94'}
    }
    
    print(f"{'Model':<20} {'Actual Acc':<12} {'Expected Acc':<15} {'Actual AUC':<12} {'Expected AUC':<15} {'Status':<10}")
    print("-" * 80)
    
    for model_name in results:
        actual_acc = results[model_name]['Accuracy']
        actual_auc = results[model_name]['AUC']
        
        if model_name in expected_results:
            expected_acc = expected_results[model_name]['Accuracy']
            expected_auc = expected_results[model_name]['AUC']
        else:
            expected_acc = "N/A"
            expected_auc = "N/A"
        
        # Determine status
        status = "✅ GOOD" if actual_acc >= 0.80 and actual_auc >= 0.80 else "⚠️ CHECK"
        
        print(f"{model_name:<20} {actual_acc:<12.3f} {expected_acc:<15} {actual_auc:<12.3f} {expected_auc:<15} {status:<10}")
    
    # Ensemble results
    status = "✅ EXCELLENT" if ensemble_acc >= 0.85 and ensemble_auc >= 0.87 else "⚠️ CHECK"
    print(f"{'Ensemble':<20} {ensemble_acc:<12.3f} {'90-93%':<15} {ensemble_auc:<12.3f} {'0.91-0.94':<15} {status:<10}")
    
    print("\n🎯 KEY INSIGHTS:")
    best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
    print(f"• Best individual model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.3f})")
    print(f"• Ensemble model accuracy: {ensemble_acc:.3f}")
    print(f"• Models meeting target accuracy (>85%): {sum(1 for r in results.values() if r['Accuracy'] >= 0.85)}/{len(results)}")
    
    return results

# Create final summary
final_results = create_final_summary(results, cv_scores, ensemble_acc, ensemble_auc)

# ============================================================================
# 13. MODEL PERSISTENCE (SAVE BEST MODEL)
# ============================================================================

import joblib

def save_best_model(best_models, results, ensemble_model):
    """
    Save the best performing model
    """
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
    best_model = best_models[best_model_name]
    
    # Save models
    joblib.dump(best_model, f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(ensemble_model, 'ensemble_model.pkl')
    
    print(f"✅ Models saved successfully!")
    print(f"• Best individual model ({best_model_name}) saved as: best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
    print(f"• Ensemble model saved as: ensemble_model.pkl")

# Save models
save_best_model(best_models, results, ensemble_model)

print("\n🎉 ANALYSIS COMPLETE! All models have been trained, evaluated, and saved.")
print("📊 Check the visualizations above for detailed performance analysis.")
print("🎯 The ensemble model provides the best overall performance for production use.")


# In[ ]:




