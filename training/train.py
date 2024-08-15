import pandas as pd
import numpy as np
import json
import logging
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from cuml.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import sklearn.metrics as metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_FILES = [
    "cicids2017/Monday-WorkingHours.csv",
    "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv",
    "cicids2017/Wednesday-workingHours.pcap_ISCX.csv",
    "cicids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "cicids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "cicids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

FEATURES = [
    "Init_Win_bytes_forward",
    "Destination Port",
    "Packet Length Variance",
    "Average Packet Size",
    "Packet Length Std",
    "Max Packet Length",
    "Subflow Fwd Bytes",
    "Bwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd Packet Length Min",
    "Bwd Packet Length Std",
    "Bwd Packet Length Min",
    "Init_Win_bytes_backward",
    "Fwd Packet Length Std",
    "Packet Length Mean",
    "Fwd Header Length",
    "Fwd Packet Length Max",
    "Fwd Header Length.1",
    "Bwd Header Length",
]


def calculate_distribution(df, label_column="Label"):
    """Calculate the distribution of traffic types."""
    distribution = df[label_column].value_counts(normalize=True).to_dict()
    distribution = {
        key: round(val * 100, 2) for key, val in distribution.items()
    }
    return distribution


def load_and_preprocess_data(file_paths):
    logger.info(f"Loading data from files: {file_paths}")

    df_list = [pd.read_csv(file_path) for file_path in file_paths]
    df = pd.concat(df_list, ignore_index=True)

    logger.info(f"Combined DataFrame shape: {df.shape}")

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

    # Calculate distribution before undersampling
    distribution_before = calculate_distribution(df)
    with open("distribution_before.json", "w") as f:
        json.dump(distribution_before, f, indent=4)

    # Undersample the majority class
    df_majority = df[df["Label"] == "BENIGN"]
    df_minority = df[df["Label"] != "BENIGN"]

    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    logger.info(f"Balanced DataFrame shape: {df_balanced.shape}")

    # Calculate distribution after undersampling
    distribution_after = calculate_distribution(df_balanced)
    with open("distribution_after.json", "w") as f:
        json.dump(distribution_after, f, indent=4)

    string_features = df_balanced.select_dtypes(include=["object"]).columns.tolist()
    string_features.remove("Label")
    for feature in string_features:
        df_balanced[feature] = pd.factorize(df_balanced[feature])[0]

    df_balanced["Label"] = df_balanced["Label"].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    X = df_balanced[FEATURES]
    y = df_balanced["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_cp = cp.asarray(X_scaled)
    y_cp = cp.asarray(y.values)

    return X_cp, y_cp, scaler


def train_and_evaluate_models(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_model = cuRF(n_estimators=50, random_state=42, n_streams=1)
    knn_model = cuKNN(n_neighbors=5)
    lr_model = LogisticRegression(max_iter=300, random_state=42)
    svm_model = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    nb_model = GaussianNB()

    models = {
        "RandomForest": rf_model,
        "KNN": knn_model,
        "LogisticRegression": lr_model,
        "SVM": svm_model,
        "MLP": mlp_model,
        "NaiveBayes": nb_model,
    }

    results = {}

    for model_name, model in models.items():
        logger.info(f"Training {model_name}")
        model_predictions = []
        y_tests = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1} for {model_name}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Convert CuPy arrays to NumPy arrays for models that require it
            if model_name in ["LogisticRegression", "MLP", "NaiveBayes"]:
                X_train = cp.asnumpy(X_train)
                X_test = cp.asnumpy(X_test)
                y_train = cp.asnumpy(y_train)
                y_test = cp.asnumpy(y_test)

            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                pred = model.predict(X_test)
            else:
                pred = model.predict(X_test).get()

            model_predictions.extend(pred.tolist())
            y_tests.extend(y_test.tolist())

        report = classification_report(y_tests, model_predictions, output_dict=True)
        cm = confusion_matrix(y_tests, model_predictions)

        logger.info(f"{model_name} Model Results:")
        print_model_metrics(y_tests, model_predictions)
        print("Confusion Matrix:")
        print(cm)

        results[model_name] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        with open(f"{model_name.lower()}_classification_report.json", "w") as f:
            json.dump(report, f, indent=4)

        with open(f"{model_name.lower()}_confusion_matrix.json", "w") as f:
            json.dump(cm.tolist(), f, indent=4)

        joblib.dump(model, f"{model_name.lower()}_model.pkl")

    return results


def print_model_metrics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 = {f1}")


X, y, scaler = load_and_preprocess_data(CSV_FILES)
if X is not None and y is not None:
    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    results = train_and_evaluate_models(X, y)
    joblib.dump(scaler, "scaler.pkl")
