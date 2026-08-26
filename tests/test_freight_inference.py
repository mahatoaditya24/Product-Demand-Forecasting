"""
Unit Tests for Freight Cost Prediction Inference Pipeline.
"""

import unittest
import pandas as pd
from Inference.Predict_freight import predict_freight_cost


class TestFreightInference(unittest.TestCase):
    """Test suite for freight cost estimation model."""

    def test_single_freight_prediction(self):
        """Verify freight cost calculation returns positive numeric result."""
        sample_data = {
            "Quantity": [1200],
            "Dollars": [18500.0]
        }
        res_df = predict_freight_cost(sample_data)
        self.assertIn("Predict_Freight", res_df.columns)
        self.assertGreater(float(res_df["Predict_Freight"][0]), 0.0)

    def test_batch_freight_prediction(self):
        """Verify multiple shipment records are batch processed correctly."""
        sample_data = {
            "Quantity": [50, 100, 500, 1200],
            "Dollars": [500.0, 1500.0, 8000.0, 18500.0]
        }
        res_df = predict_freight_cost(sample_data)
        self.assertEqual(len(res_df), 4)
        for val in res_df["Predict_Freight"]:
            self.assertGreaterEqual(float(val), 0.0)

    def test_freight_monotonicity_on_volume(self):
        """Higher quantity shipments should generally incur higher total freight."""
        small_shipment = predict_freight_cost({"Quantity": [10], "Dollars": [200.0]})
        large_shipment = predict_freight_cost({"Quantity": [2000], "Dollars": [30000.0]})
        self.assertGreater(
            float(large_shipment["Predict_Freight"][0]),
            float(small_shipment["Predict_Freight"][0])
        )


if __name__ == "__main__":
    unittest.main()
