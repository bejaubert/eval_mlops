import mlflow.pyfunc
import pandas as pd
import joblib


class XGBoostWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper MLflow pour un modèle XGBoost.
    """

    def load_context(self, context):
        """
        Chargé automatiquement lors du chargement avec pyfunc.load_model.
        Il permet de récupérer les artefacts du modèle.

        Args:
            context (mlflow.pyfunc.PythonModelContext): contexte MLflow
        """
        model_path = context.artifacts["model_path"]
        self.model = joblib.load(model_path)

    def predict(self, context, model_input):
        """
        Prédiction à partir d'un DataFrame en entrée.

        Args:
            context: non utilisé ici.
            model_input (pd.DataFrame): données d’entrée (colonnes = features)

        Returns:
            List[int]: prédictions sous forme de liste
        """
        return self.model.predict(model_input).tolist()
