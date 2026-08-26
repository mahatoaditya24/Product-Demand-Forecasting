"""
Unit Tests for Vendor Invoice Risk Classification Inference Pipeline.
"""

import unittest
import pandas as pd
from Inference.predict_invoice_flag import predict_invoice_flag


class TestInvoiceInference(unittest.TestCase):
    """Test suite for vendor invoice risk audit classification."""

    def test_invoice_flag_prediction_binary_output(self):
        """Verify model returns a binary flag (0 or 1)."""
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

    def test_high_discrepancy_invoice_flagged(self):
        """Massive dollar discrepancies between invoice and PO should flag for review."""
        suspicious_data = {
            "invoice_quantity": [500],
            "invoice_dollars": [45000.0],
            "Freight": [150.0],
            "total_item_quantity": [10],
            "total_item_dollars": [500.0]
        }
        res_df = predict_invoice_flag(suspicious_data)
        self.assertEqual(int(res_df["Predict_Flag"][0]), 1)

    def test_batch_invoice_processing(self):
        """Multiple invoices should process in batch without index misalignment."""
        batch_data = {
            "invoice_quantity": [50, 100, 25],
            "invoice_dollars": [352.95, 1200.0, 180.0],
            "Freight": [1.73, 15.0, 2.0],
            "total_item_quantity": [162, 100, 25],
            "total_item_dollars": [2476.0, 1200.0, 180.0]
        }
        res_df = predict_invoice_flag(batch_data)
        self.assertEqual(len(res_df), 3)


if __name__ == "__main__":
    unittest.main()
