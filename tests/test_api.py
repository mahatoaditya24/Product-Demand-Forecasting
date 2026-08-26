"""
Unit Tests for FastAPI REST Microservice Endpoints & Pydantic Validation.
"""

import unittest

try:
    from fastapi.testclient import TestClient
    from api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI / TestClient not installed in local environment")
class TestSupplyChainAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def setUp(self):
        if FASTAPI_AVAILABLE:
            self.client = TestClient(app)

    def test_root_endpoint(self):
        """Verify API root endpoint returns system metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OPERATIONAL")
        self.assertIn("active_modules", data)

    def test_health_check(self):
        """Verify health check returns HEALTHY status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")

    def test_metadata_warehouses(self):
        """Verify warehouse metadata endpoint returns 4 regional centers."""
        response = self.client.get("/api/v1/metadata/warehouses")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Whse_J", data["warehouses"])
        self.assertEqual(len(data["warehouses"]), 4)

    def test_predict_freight_endpoint(self):
        """Verify POST /api/v1/predict/freight returns valid cost forecast."""
        payload = {
            "quantity": 1200,
            "dollars": 18500.0
        }
        response = self.client.post("/api/v1/predict/freight", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["predicted_freight_cost"], 0.0)
        self.assertEqual(data["status"], "SUCCESS")

    def test_predict_invoice_risk_endpoint(self):
        """Verify POST /api/v1/predict/invoice-risk returns risk classification."""
        payload = {
            "invoice_quantity": 50,
            "invoice_dollars": 352.95,
            "freight": 1.73,
            "total_item_quantity": 162,
            "total_item_dollars": 2476.0
        }
        response = self.client.post("/api/v1/predict/invoice-risk", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_flagged", data)
        self.assertIn("risk_tier", data)
        self.assertIn("recommendation", data)

    def test_predict_demand_endpoint(self):
        """Verify POST /api/v1/predict/demand returns SKU forecast and safety stock."""
        payload = {
            "product_code": "Product_0993",
            "warehouse": "Whse_J",
            "product_category": "Category_028",
            "forecast_date": "2026-09-01",
            "lag_1": 520.0,
            "lag_7": 500.0,
            "lag_30": 480.0,
            "rolling_mean_7": 510.0,
            "rolling_std_7": 45.0
        }
        response = self.client.post("/api/v1/predict/demand", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["predicted_order_demand"], 0.0)
        self.assertGreaterEqual(data["recommended_safety_stock"], 0)
        self.assertEqual(data["product_code"], "Product_0993")

    def test_predict_demand_batch_endpoint(self):
        """Verify POST /api/v1/predict/demand/batch processes multiple SKU items."""
        payload = {
            "items": [
                {
                    "product_code": "Product_0993",
                    "warehouse": "Whse_J",
                    "product_category": "Category_028",
                    "forecast_date": "2026-09-01",
                    "lag_1": 520.0,
                    "lag_7": 500.0,
                    "lag_30": 480.0,
                    "rolling_mean_7": 510.0,
                    "rolling_std_7": 45.0
                },
                {
                    "product_code": "Product_1521",
                    "warehouse": "Whse_S",
                    "product_category": "Category_019",
                    "forecast_date": "2026-09-02",
                    "lag_1": 1200.0,
                    "lag_7": 1150.0,
                    "lag_30": 1100.0,
                    "rolling_mean_7": 1180.0,
                    "rolling_std_7": 95.0
                }
            ]
        }
        response = self.client.post("/api/v1/predict/demand/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_processed"], 2)
        self.assertEqual(len(data["predictions"]), 2)

    def test_invalid_payload_validation_error(self):
        """Verify negative values or missing fields trigger HTTP 422 Unprocessable Entity."""
        invalid_payload = {
            "quantity": -50,
            "dollars": 1000.0
        }
        response = self.client.post("/api/v1/predict/freight", json=invalid_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
