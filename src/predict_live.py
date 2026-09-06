import os
import joblib
import pandas as pd

HEALTH_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "tyre_health_model.pkl"
)
RUL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "tyre_rul_model.pkl"
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


class TyrePredictor:

    def __init__(self):

        health_package = _load_pkl(HEALTH_MODEL_PATH)
        rul_package    = _load_pkl(RUL_MODEL_PATH)

        self.health_model  = health_package["model"]
        self.label_encoder = health_package["label_encoder"]
        self.features      = health_package["features"]
        self.rul_model     = rul_package["model"]

    def predict(self, tyre_data):

        input_df = tyre_data[self.features].copy()

        health_prediction = self.health_model.predict(input_df)
        health_status     = self.label_encoder.inverse_transform(health_prediction)
        health_probability = self.health_model.predict_proba(input_df)

        remaining_life = self.rul_model.predict(input_df)

        result = tyre_data.copy()
        result["health_status"]     = health_status
        result["remaining_life_km"] = remaining_life.round(0)
        result["confidence"]        = (health_probability.max(axis=1) * 100).round(2)

        return result
