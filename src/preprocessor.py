import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame, target_col: str = "readmission"):
        df = df.copy()

        y = df[target_col]
        X = df.drop(columns=[target_col])

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled, y

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        X_scaled = self.scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=df.columns)

        return X_scaled
