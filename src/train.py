import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessor import Preprocessor
from sklearn.metrics import precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

EXPERIMENT_NAME = "hospital-readmission-xgb"


def load_data(path="data/DSA-2025_clean_data.tsv"):
    """
    Charge les données TSV nettoyées.
    """
    return pd.read_csv(path, sep="\t")


def train_model():
    """
    Lance l'entraînement avec Grid Search manuelle + log MLflow.
    """
    mlflow.set_tracking_uri("https://user-benjamin389-mlflow.user.lab.sspcloud.fr")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df, target_col="readmission")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Grid Search : combinaison de deux hyperparamètres
    max_depths = [3, 5]
    learning_rates = [0.1, 0.3]

    for max_depth in max_depths:
        for learning_rate in learning_rates:
            with mlflow.start_run():
                # Log des hyperparamètres
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("learning_rate", learning_rate)

                model = xgb.XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    use_label_encoder=False,
                    eval_metric="logloss"
                )

                model.fit(X_train, y_train)

                # Évaluation
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision_score(y_test, preds))
                mlflow.log_metric("recall", recall_score(y_test, preds))
                mlflow.log_metric("f1", f1_score(y_test, preds))
                signature = infer_signature(X_test, preds)
                input_example = X_test.iloc[:1]

                # Log du modèle en tant que modèle "sklearn"
                mlflow.sklearn.log_model(model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example)


if __name__ == "__main__":
    train_model()
