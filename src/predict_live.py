import joblib
import pandas as pd

HEALTH_MODEL_PATH = "models/tyre_health_model.pkl"
RUL_MODEL_PATH = "models/tyre_rul_model.pkl"


class TyrePredictor:

    def __init__(self):

        health_package = joblib.load(
            HEALTH_MODEL_PATH
        )

        rul_package = joblib.load(
            RUL_MODEL_PATH
        )

        self.health_model = (
            health_package["model"]
        )

        self.label_encoder = (
            health_package["label_encoder"]
        )

        self.features = (
            health_package["features"]
        )

        self.rul_model = (
            rul_package["model"]
        )

    def predict(self, tyre_data):

        input_df = tyre_data[
            self.features
        ].copy()

        # Health prediction
        health_prediction = (
            self.health_model.predict(input_df)
        )

        health_status = (
            self.label_encoder.inverse_transform(
                health_prediction
            )
        )

        health_probability = (
            self.health_model.predict_proba(
                input_df
            )
        )

        # RUL prediction
        remaining_life = (
            self.rul_model.predict(input_df)
        )

        result = tyre_data.copy()

        result["health_status"] = health_status

        result["remaining_life_km"] = (
            remaining_life.round(0)
        )

        # Maximum prediction confidence
        result["confidence"] = (
            health_probability.max(axis=1) * 100
        ).round(2)

        return result