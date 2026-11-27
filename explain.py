"""SHAP explanations for a loan-approval model.

Works with tree models (RandomForest, XGBoost, GradientBoosting).
If your final model is a Voting/Stacking ensemble, pass the best tree model inside it.
"""
import numpy as np
import shap


class LoanExplainer:
    def __init__(self, tree_model):
        self.explainer = shap.TreeExplainer(tree_model)

    def _positive_class_values(self, sv):
        # Different shap versions return different shapes
        if isinstance(sv, list):          # older: list of arrays, one per class
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:  # newer: (rows, features, classes)
            sv = sv[:, :, 1]
        return np.asarray(sv)

    def top_reasons(self, input_df, n=3):
        """Return the n features that pushed the decision the most."""
        sv = self._positive_class_values(self.explainer.shap_values(input_df))
        vals = sv[0]
        order = np.argsort(-np.abs(vals))[:n]
        out = []
        for i in order:
            out.append({
                "feature": str(input_df.columns[i]),
                "value": float(input_df.iloc[0, i]),
                "impact": float(vals[i]),
                "effect": "increased approval chance" if vals[i] > 0 else "decreased approval chance",
            })
        return out
