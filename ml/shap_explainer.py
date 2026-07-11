import numpy as np
import shap

class ShapExplainer:
    def __init__(self, model, feature_order: list):
        self.model = model
        self.feature_order = feature_order

        self.explainer = shap.TreeExplainer(model)

    def explain(self, feature_vector: np.ndarray, prediction_id: int) -> dict:
        raw_shap_output = self.explainer.shap_values(feature_vector)

      
        if isinstance(raw_shap_output, list):
            values_for_class = raw_shap_output[prediction_id][0]
        elif raw_shap_output.ndim == 3:
            values_for_class = raw_shap_output[0, :, prediction_id]
        else:
            values_for_class = raw_shap_output[0]

        attribution = {}
        for feature_name, shap_value in zip(self.feature_order, values_for_class):
            attribution[feature_name] = {
                "shap_value": round(float(shap_value), 4),
                "direction": "pushed toward approval" if shap_value > 0 else "pushed toward review",
            }

        return attribution