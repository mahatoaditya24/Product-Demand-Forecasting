"""
Integration & Regression Inference Tests for Machine Learning Models.
"""

import unittest
import pandas as pd
from Inference.Predict_freight import predict_freight_cost
from Inference.predict_invoice_flag import predict_invoice_flag
from Inference.predict_demand import (
    prepare_demand_features,
    predict_product_demand
)


class TestModelInference(unittest.TestCase):
    """End-to-end integration tests for all 3 machine learning models."""

    def test_freight_prediction_pipeline(self):
        """Test freight cost prediction output."""
        sample_data = {
            "Quantity": [100, 50, 10, 25, 80],
            "Dollars": [18500, 9000, 300, 2500, 7000]
        }
        res_df = predict_freight_cost(sample_data)
        self.assertIn("Predict_Freight", res_df.columns)
        self.assertEqual(len(res_df), 5)

    def test_invoice_flagging_pipeline(self):
        """Test invoice risk classification output."""
        sample_data = {
            "invoice_quantity": [50],
            "invoice_dollars": [352.95],
            "Freight": [1.73],
            "total_item_quantity": [162],
            "total_item_dollars": [2476.0]
        }
        res_df = predict_invoice_flag(sample_data)
        self.assertIn("Predict_Flag", res_df.columns)
        self.assertIn(int(res_df["Predict_Flag"][0]), [0, 1])

    def test_demand_forecasting_pipeline(self):
        """Test product demand forecasting output."""
        features_df = prepare_demand_features(
            product_code="Product_0993",
            warehouse="Whse_J",
            product_category="Category_028",
            forecast_date="2026-09-01",
            lag_1=520.0,
            lag_7=500.0,
            lag_30=480.0,
            rolling_mean_7=510.0,
            rolling_std_7=45.0
        )
        prediction = predict_product_demand(features_df)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)


if __name__ == "__main__":
    unittest.main()
