"""
Pydantic Schemas for Supply Chain & Vendor Intelligence API.
Enforces request validation, boundary constraints, and type-safe response models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Freight Cost Models
# -----------------------------------------------------------------------------
class FreightPredictionRequest(BaseModel):
    quantity: int = Field(..., gt=0, description="Total units shipped", example=1200)
    dollars: float = Field(..., gt=0.0, description="Invoice dollar amount", example=18500.0)


class FreightPredictionResponse(BaseModel):
    predicted_freight_cost: float = Field(..., description="Estimated freight cost in USD", example=342.50)
    freight_to_invoice_ratio_pct: float = Field(..., description="Freight as percentage of invoice amount", example=1.85)
    cost_per_unit: float = Field(..., description="Shipping cost per unit", example=0.29)
    status: str = Field(default="SUCCESS")


# -----------------------------------------------------------------------------
# Invoice Risk Models
# -----------------------------------------------------------------------------
class InvoiceRiskRequest(BaseModel):
    invoice_quantity: int = Field(..., gt=0, description="Quantity declared on vendor invoice", example=50)
    invoice_dollars: float = Field(..., gt=0.0, description="Dollar total on vendor invoice", example=352.95)
    freight: float = Field(..., ge=0.0, description="Freight fee on invoice", example=1.73)
    total_item_quantity: int = Field(..., gt=0, description="Quantity recorded in PO/receipt", example=162)
    total_item_dollars: float = Field(..., gt=0.0, description="Dollar valuation of PO line items", example=2476.0)


class InvoiceRiskResponse(BaseModel):
    is_flagged: bool = Field(..., description="True if invoice requires manual approval audit", example=True)
    flag_code: int = Field(..., description="Binary flag code (0: Approved, 1: Flagged)", example=1)
    risk_tier: str = Field(..., description="Risk category: APPROVED, WARNING, or CRITICAL", example="CRITICAL")
    dollar_discrepancy: float = Field(..., description="Absolute variance between invoice and PO dollars", example=2123.05)
    quantity_discrepancy: int = Field(..., description="Variance between invoice and received units", example=112)
    recommendation: str = Field(..., description="Actionable finance guidance", example="Hold payment. Discrepancy exceeds audit threshold.")


# -----------------------------------------------------------------------------
# Product Demand Forecasting Models
# -----------------------------------------------------------------------------
class DemandPredictionRequest(BaseModel):
    product_code: str = Field(..., description="SKU identifier (e.g. Product_0993)", example="Product_0993")
    warehouse: str = Field(..., description="Target warehouse code (Whse_A, Whse_C, Whse_J, Whse_S)", example="Whse_J")
    product_category: str = Field(..., description="Product category (e.g. Category_028)", example="Category_028")
    forecast_date: str = Field(..., description="Target forecast date (YYYY-MM-DD)", example="2026-09-01")
    lag_1: float = Field(..., ge=0.0, description="Demand from previous day", example=520.0)
    lag_7: float = Field(..., ge=0.0, description="Demand from 7 days ago", example=500.0)
    lag_30: float = Field(..., ge=0.0, description="Demand from 30 days ago", example=480.0)
    rolling_mean_7: float = Field(..., ge=0.0, description="7-day rolling average demand", example=510.0)
    rolling_std_7: float = Field(..., ge=0.0, description="7-day rolling standard deviation", example=45.0)


class DemandPredictionResponse(BaseModel):
    product_code: str
    warehouse: str
    forecast_date: str
    predicted_order_demand: float = Field(..., description="Predicted unit demand", example=515.20)
    recommended_safety_stock: int = Field(..., description="Buffer inventory recommended (95% service level)", example=589)
    status: str = Field(default="SUCCESS")


class BatchDemandItem(BaseModel):
    product_code: str
    warehouse: str
    product_category: str
    forecast_date: str
    lag_1: float
    lag_7: float
    lag_30: float
    rolling_mean_7: float
    rolling_std_7: float


class BatchDemandRequest(BaseModel):
    items: List[BatchDemandItem]


class BatchDemandResponse(BaseModel):
    total_processed: int
    predictions: List[DemandPredictionResponse]
