import os
import joblib
import pandas as pd

ENV_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "tyre_env_model.pkl"
)


def _load_pkl(path: str):
    """Load a joblib pickle, raising a clear error if it is an LFS pointer."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run `git lfs pull` to download the model files."
        )
    size = os.path.getsize(path)
    if size < 512:          # LFS pointer files are ~130 bytes
        raise RuntimeError(
            f"Model file looks like a Git LFS pointer (size={size} bytes): {path}\n"
            "Run `git lfs pull` to download the actual model files."
        )
    return joblib.load(path)


class EnvPredictor:

    def __init__(self):
        pkg = _load_pkl(ENV_MODEL_PATH)
        self.model       = pkg["model"]
        self.features    = pkg["features"]
        self.targets     = pkg["targets"]
        self.weather_map = pkg["weather_map"]
        self.road_map    = pkg["road_map"]

    def predict(self, tyre_data):

        df = tyre_data.copy()

        df["weather_encoded"] = df["weather_condition"].map(self.weather_map).fillna(0)
        df["road_encoded"]    = df.get(
            "road_condition", pd.Series("GOOD", index=df.index)
        ).map(self.road_map).fillna(0)

        if "wear_ratio" not in df.columns:
            df["wear_ratio"] = (df["tyre_age_km"] / 60000).clip(0, 1)

        input_df = df[self.features]
        preds    = self.model.predict(input_df)

        result = tyre_data.copy()
        for i, col in enumerate(self.targets):
            result[col] = preds[:, i].round(5)

        return result
