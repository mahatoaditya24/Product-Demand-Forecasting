from pathlib import Path
import datetime
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Notebook" / "best_demand_forecasting_model.pkl"

# Fallback warehouse and category mappings
WAREHOUSES = ["Whse_A", "Whse_C", "Whse_J", "Whse_S"]
CATEGORIES = [f"Category_{i:03d}" for i in range(1, 34)]


def load_demand_model(model_path=MODEL_PATH):
    """Load trained demand forecasting model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Demand forecasting model not found at {model_path}")
    return joblib.load(model_path)


def extract_numeric_code(text_val: str, prefix: str = "") -> int:
    """Safely extract integer index from category string like 'Product_0993' or 'Category_028'."""
    try:
        clean_text = text_val.replace(prefix, "").strip("_")
        return int(clean_text)
    except Exception:
        return 0


def prepare_demand_features(
    product_code: str,
    warehouse: str,
    product_category: str,
    forecast_date: datetime.date,
    lag_1: float,
    lag_7: float,
    lag_30: float,
    rolling_mean_7: float,
    rolling_std_7: float
) -> pd.DataFrame:
    """Transform user inputs into the exact 17-feature tabular structure expected by the model."""
    
    # 1. Categorical Encoding (deterministic ordinal mapping)
    whse_code = WAREHOUSES.index(warehouse) if warehouse in WAREHOUSES else 0
    cat_code = extract_numeric_code(product_category, "Category")
    prod_code = extract_numeric_code(product_code, "Product")

    # 2. Calendar / Time Features
    year = forecast_date.year
    month = forecast_date.month
    day = forecast_date.day
    weekday = forecast_date.weekday()
    quarter = (forecast_date.month - 1) // 3 + 1

    # 3. Cyclical Trigonometric Features
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    weekday_sin = np.sin(2 * np.pi * weekday / 7.0)
    weekday_cos = np.cos(2 * np.pi * weekday / 7.0)

    # 4. Feature DataFrame in exact training order
    feature_dict = {
        "Product_Code": [prod_code],
        "Warehouse": [whse_code],
        "Product_Category": [cat_code],
        "Year": [year],
        "Month": [month],
        "Weekday": [weekday],
        "Quarter": [quarter],
        "Day": [day],
        "Month_sin": [month_sin],
        "Month_cos": [month_cos],
        "Weekday_sin": [weekday_sin],
        "Weekday_cos": [weekday_cos],
        "lag_1": [float(lag_1)],
        "lag_7": [float(lag_7)],
        "lag_30": [float(lag_30)],
        "rolling_mean_7": [float(rolling_mean_7)],
        "rolling_std_7": [float(rolling_std_7)],
    }

    df = pd.DataFrame(feature_dict)
    # Ensure column order matches model if feature_names_in_ is present
    try:
        model = load_demand_model()
        if hasattr(model, "feature_names_in_"):
            df = df[model.feature_names_in_]
    except Exception:
        pass

    return df


def predict_product_demand(features_df: pd.DataFrame) -> float:
    """Perform model inference and return the predicted order demand."""
    model = load_demand_model()
    prediction = model.predict(features_df)
    predicted_val = float(prediction[0])
    # Demand cannot be negative
    return max(0.0, round(predicted_val, 2))


if __name__ == "__main__":
    test_features = prepare_demand_features(
        product_code="Product_0993",
        warehouse="Whse_J",
        product_category="Category_028",
        forecast_date=datetime.date(2016, 5, 20),
        lag_1=500.0,
        lag_7=450.0,
        lag_30=400.0,
        rolling_mean_7=480.0,
        rolling_std_7=35.0
    )
    predicted_demand = predict_product_demand(test_features)
    print("Predicted Demand:", predicted_demand)
