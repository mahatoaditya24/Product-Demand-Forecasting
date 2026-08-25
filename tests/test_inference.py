import datetime
import pandas as pd
import pytest
from Inference.Predict_freight import predict_freight_cost
from Inference.predict_invoice_flag import predict_invoice_flag
from Inference.predict_demand import (
    prepare_demand_features,
    predict_product_demand,
    predict_batch_demand,
    generate_sample_batch_csv
)


def test_freight_prediction():
    """Verify that freight cost regression model predicts positive valid cost."""
    sample_data = {
        "Quantity": [100, 50],
        "Dollars": [5000, 2500]
    }
    result_df = predict_freight_cost(sample_data)
    assert "Predict_Freight" in result_df.columns
    assert len(result_df) == 2
    assert result_df["Predict_Freight"].iloc[0] > 0
    assert result_df["Predict_Freight"].iloc[1] > 0


def test_invoice_flag_prediction():
    """Verify that invoice risk classifier returns binary flag (0 or 1)."""
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0]
    }
    result_df = predict_invoice_flag(sample_data)
    assert "Predict_Flag" in result_df.columns
    assert result_df["Predict_Flag"].iloc[0] in [0, 1]


def test_demand_prediction_single():
    """Verify that single SKU demand forecasting generates valid non-negative demand."""
    features_df = prepare_demand_features(
        product_code="Product_0993",
        warehouse="Whse_J",
        product_category="Category_028",
        forecast_date=datetime.date(2026, 9, 1),
        lag_1=500.0,
        lag_7=480.0,
        lag_30=450.0,
        rolling_mean_7=490.0,
        rolling_std_7=45.0
    )
    assert len(features_df.columns) == 17
    predicted_units = predict_product_demand(features_df)
    assert isinstance(predicted_units, float)
    assert predicted_units >= 0.0


def test_demand_prediction_batch():
    """Verify that batch CSV forecasting correctly processes multi-row DataFrames."""
    sample_df = generate_sample_batch_csv()
    assert len(sample_df) == 5
    result_df = predict_batch_demand(sample_df)
    assert "Predicted_Order_Demand" in result_df.columns
    assert "Recommended_Safety_Stock" in result_df.columns
    assert (result_df["Predicted_Order_Demand"] >= 0).all()
    assert (result_df["Recommended_Safety_Stock"] >= result_df["Predicted_Order_Demand"]).all()
