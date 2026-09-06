import joblib
import pandas as pd

ENV_MODEL_PATH = "models/tyre_env_model.pkl"


class EnvPredictor:

    def __init__(self):
        pkg = joblib.load(ENV_MODEL_PATH)
        self.model       = pkg["model"]
        self.features    = pkg["features"]
        self.targets     = pkg["targets"]
        self.weather_map = pkg["weather_map"]
        self.road_map    = pkg["road_map"]

    def predict(self, tyre_data):

        df = tyre_data.copy()

        df["weather_encoded"] = df["weather_condition"].map(self.weather_map).fillna(0)
        df["road_encoded"]    = df.get("road_condition", pd.Series("GOOD", index=df.index)).map(self.road_map).fillna(0)

        # wear_ratio derived if not present
        if "wear_ratio" not in df.columns:
            df["wear_ratio"] = (df["tyre_age_km"] / 60000).clip(0, 1)

        input_df = df[self.features]
        preds    = self.model.predict(input_df)

        result = tyre_data.copy()
        for i, col in enumerate(self.targets):
            result[col] = preds[:, i].round(5)

        return result
