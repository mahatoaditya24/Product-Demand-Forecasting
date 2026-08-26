"""
Unit Tests for SKU Demand Forecasting & Feature Engineering Pipeline.
"""

import unittest
import datetime
import pandas as pd
import numpy as np
from Inference.predict_demand import (
    prepare_demand_features,
    predict_product_demand,
    predict_batch_demand,
    extract_numeric_code,
    WAREHOUSES,
    CATEGORIES
)


class TestDemandInference(unittest.TestCase):
    """Test suite for product demand time-series tabular regression."""

    def test_extract_numeric_code(self):
        """Verify numeric code extraction from string identifiers."""
        self.assertEqual(extract_numeric_code("Product_0993", "Product"), 993)
        self.assertEqual(extract_numeric_code("Category_028", "Category"), 28)
        self.assertEqual(extract_numeric_code("Unknown", "Product"), 0)

    def test_prepare_demand_features_shape(self):
        """Verify prepared feature dataframe contains all required temporal and lag features."""
        feat_df = prepare_demand_features(
            product_code="Product_0993",
            warehouse="Whse_J",
            product_category="Category_028",
            forecast_date=datetime.date(2026, 9, 1),
            lag_1=520.0,
            lag_7=500.0,
            lag_30=480.0,
            rolling_mean_7=510.0,
            rolling_std_7=45.0
        )
        self.assertEqual(len(feat_df), 1)
        expected_cols = [
            "Product_Code", "Warehouse", "Product_Category",
            "Year", "Month", "Weekday", "Quarter", "Day",
            "Month_sin", "Month_cos", "Weekday_sin", "Weekday_cos",
            "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_std_7"
        ]
        for col in expected_cols:
            self.assertIn(col, feat_df.columns)

    def test_single_demand_prediction(self):
        """Verify model returns a non-negative order demand estimate."""
        feat_df = prepare_demand_features(
            product_code="Product_0993",
            warehouse="Whse_J",
            product_category="Category_028",
            forecast_date=datetime.date(2026, 9, 1),
            lag_1=520.0,
            lag_7=500.0,
            lag_30=480.0,
            rolling_mean_7=510.0,
            rolling_std_7=45.0
        )
        pred = predict_product_demand(feat_df)
        self.assertIsInstance(pred, float)
        self.assertGreaterEqual(pred, 0.0)

    def test_batch_demand_prediction(self):
        """Verify batch DataFrame processing outputs predictions and safety stocks."""
        input_data = {
            "Product_Code": ["Product_0993", "Product_1521"],
            "Warehouse": ["Whse_J", "Whse_S"],
            "Product_Category": ["Category_028", "Category_019"],
            "Date": ["2026-09-01", "2026-09-02"],
            "lag_1": [520.0, 1200.0],
            "lag_7": [500.0, 1150.0],
            "lag_30": [480.0, 1100.0],
            "rolling_mean_7": [510.0, 1180.0],
            "rolling_std_7": [45.0, 95.0]
        }
        res_df = predict_batch_demand(pd.DataFrame(input_data))
        self.assertEqual(len(res_df), 2)
        self.assertIn("Predicted_Order_Demand", res_df.columns)
        self.assertIn("Recommended_Safety_Stock", res_df.columns)
        for row in res_df["Predicted_Order_Demand"]:
            self.assertGreaterEqual(float(row), 0.0)


if __name__ == "__main__":
    unittest.main()
