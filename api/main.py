"""
FastAPI Microservice: Supply Chain & Vendor Intelligence API.
Exposes high-performance RESTful endpoints for Freight Cost Estimation,
Vendor Invoice Risk Auditing, and SKU Demand Forecasting.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
INFERENCE_DIR = BASE_DIR / "Inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    FreightPredictionRequest,
    FreightPredictionResponse,
    InvoiceRiskRequest,
    InvoiceRiskResponse,
    DemandPredictionRequest,
    DemandPredictionResponse,
    BatchDemandRequest,
    BatchDemandResponse,
    BatchDemandItem
)

from Inference.Predict_freight import predict_freight_cost
from Inference.predict_invoice_flag import predict_invoice_flag
from Inference.predict_demand import (
    prepare_demand_features,
    predict_product_demand,
    WAREHOUSES,
    CATEGORIES
)

# Initialize FastAPI Application
app = FastAPI(
    title="Supply Chain & Vendor Intelligence API",
    description="Enterprise Machine Learning Serving Microservice for Freight Estimation, Invoice Audit Flagging, and Multi-Warehouse SKU Demand Forecasting.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root() -> Dict[str, Any]:
    """Root metadata endpoint with system information and active modules."""
    return {
        "system": "Supply Chain & Vendor Intelligence Platform",
        "version": "2.0.0",
        "status": "OPERATIONAL",
        "docs_url": "/docs",
        "active_modules": [
            "Freight Cost Prediction (Regression)",
            "Invoice Risk Assessment (Classification)",
            "Product Demand Forecasting (Time-Series Tabular Regression)"
        ]
    }


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Health check endpoint for Docker container orchestrators and load balancers."""
    return {
        "status": "HEALTHY",
        "models_loaded": "TRUE",
        "service": "supply-chain-api"
    }


@app.get("/api/v1/metadata/warehouses", tags=["Metadata"])
def get_warehouse_metadata() -> Dict[str, Any]:
    """Returns available regional fulfillment centers and product category lists."""
    return {
        "warehouses": WAREHOUSES,
        "total_categories": len(CATEGORIES),
        "sample_categories": CATEGORIES[:10]
    }


# -----------------------------------------------------------------------------
# 1. Freight Cost Prediction Endpoint
# -----------------------------------------------------------------------------
@app.post(
    "/api/v1/predict/freight",
    response_model=FreightPredictionResponse,
    tags=["Freight Cost Prediction"]
)
def predict_freight(request: FreightPredictionRequest):
    """Forecast transportation and freight spend based on shipment volume and invoice valuation."""
    try:
        input_data = {
            "Quantity": [request.quantity],
            "Dollars": [request.dollars]
        }
        res_df = predict_freight_cost(input_data)
        freight_cost = float(res_df["Predict_Freight"][0])

        ratio_pct = round((freight_cost / request.dollars) * 100.0, 2)
        cost_per_unit = round(freight_cost / request.quantity, 2)

        return FreightPredictionResponse(
            predicted_freight_cost=freight_cost,
            freight_to_invoice_ratio_pct=ratio_pct,
            cost_per_unit=cost_per_unit,
            status="SUCCESS"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Freight prediction inference failed: {str(e)}"
        )


# -----------------------------------------------------------------------------
# 2. Invoice Risk Assessment Endpoint
# -----------------------------------------------------------------------------
@app.post(
    "/api/v1/predict/invoice-risk",
    response_model=InvoiceRiskResponse,
    tags=["Invoice Risk Auditing"]
)
def predict_invoice_risk(request: InvoiceRiskRequest):
    """Audit vendor invoice against PO lines and classify whether manual approval is required."""
    try:
        input_data = {
            "invoice_quantity": [request.invoice_quantity],
            "invoice_dollars": [request.invoice_dollars],
            "Freight": [request.freight],
            "total_item_quantity": [request.total_item_quantity],
            "total_item_dollars": [request.total_item_dollars]
        }
        res_df = predict_invoice_flag(input_data)
        flag_val = int(res_df["Predict_Flag"][0])
        is_flagged = bool(flag_val == 1)

        dollar_diff = round(abs(request.invoice_dollars - request.total_item_dollars), 2)
        qty_diff = abs(request.invoice_quantity - request.total_item_quantity)

        if is_flagged:
            risk_tier = "CRITICAL" if dollar_diff > 500 else "WARNING"
            recommendation = f"Hold payment. Discrepancy of ${dollar_diff:,.2f} detected between PO and invoice."
        else:
            risk_tier = "APPROVED"
            recommendation = "Invoice matches purchase order tolerances. Straight-through approval granted."

        return InvoiceRiskResponse(
            is_flagged=is_flagged,
            flag_code=flag_val,
            risk_tier=risk_tier,
            dollar_discrepancy=dollar_diff,
            quantity_discrepancy=qty_diff,
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invoice risk classification failed: {str(e)}"
        )


# -----------------------------------------------------------------------------
# 3. Product Demand Forecasting Endpoint
# -----------------------------------------------------------------------------
@app.post(
    "/api/v1/predict/demand",
    response_model=DemandPredictionResponse,
    tags=["Demand Forecasting"]
)
def predict_demand(request: DemandPredictionRequest):
    """Predict future SKU demand and calculate optimal safety stock buffer for a specific warehouse."""
    try:
        features_df = prepare_demand_features(
            product_code=request.product_code,
            warehouse=request.warehouse,
            product_category=request.product_category,
            forecast_date=request.forecast_date,
            lag_1=request.lag_1,
            lag_7=request.lag_7,
            lag_30=request.lag_30,
            rolling_mean_7=request.rolling_mean_7,
            rolling_std_7=request.rolling_std_7
        )

        demand_pred = predict_product_demand(features_df)
        safety_stock = int(round(demand_pred + (1.65 * request.rolling_std_7)))

        return DemandPredictionResponse(
            product_code=request.product_code,
            warehouse=request.warehouse,
            forecast_date=request.forecast_date,
            predicted_order_demand=demand_pred,
            recommended_safety_stock=safety_stock,
            status="SUCCESS"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demand forecasting inference failed: {str(e)}"
        )


@app.post(
    "/api/v1/predict/demand/batch",
    response_model=BatchDemandResponse,
    tags=["Demand Forecasting"]
)
def predict_demand_batch(request: BatchDemandRequest):
    """Execute high-volume batch SKU demand predictions across multiple warehouses."""
    try:
        results = []
        for item in request.items:
            features_df = prepare_demand_features(
                product_code=item.product_code,
                warehouse=item.warehouse,
                product_category=item.product_category,
                forecast_date=item.forecast_date,
                lag_1=item.lag_1,
                lag_7=item.lag_7,
                lag_30=item.lag_30,
                rolling_mean_7=item.rolling_mean_7,
                rolling_std_7=item.rolling_std_7
            )
            demand_pred = predict_product_demand(features_df)
            safety_stock = int(round(demand_pred + (1.65 * item.rolling_std_7)))

            results.append(DemandPredictionResponse(
                product_code=item.product_code,
                warehouse=item.warehouse,
                forecast_date=item.forecast_date,
                predicted_order_demand=demand_pred,
                recommended_safety_stock=safety_stock,
                status="SUCCESS"
            ))

        return BatchDemandResponse(
            total_processed=len(results),
            predictions=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch demand forecasting failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
