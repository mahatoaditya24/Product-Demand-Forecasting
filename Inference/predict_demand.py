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
        clean_text = str(text_val).replace(prefix, "").strip("_")
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
    if isinstance(forecast_date, str):
        forecast_date = pd.to_datetime(forecast_date).date()
    elif isinstance(forecast_date, pd.Timestamp):
        forecast_date = forecast_date.date()

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
    return max(0.0, round(predicted_val, 2))


def predict_batch_demand(input_df: pd.DataFrame) -> pd.DataFrame:
    """Perform batch demand forecasting on an uploaded pandas DataFrame."""
    output_df = input_df.copy()
    feature_rows = []

    for _, row in input_df.iterrows():
        prod = str(row.get("Product_Code", "Product_0001"))
        whse = str(row.get("Warehouse", "Whse_J"))
        cat = str(row.get("Product_Category", "Category_001"))
        f_date = row.get("Date", datetime.date.today())
        l1 = float(row.get("lag_1", 100))
        l7 = float(row.get("lag_7", 100))
        l30 = float(row.get("lag_30", 100))
        rm7 = float(row.get("rolling_mean_7", 100))
        rs7 = float(row.get("rolling_std_7", 10))

        feat_df = prepare_demand_features(prod, whse, cat, f_date, l1, l7, l30, rm7, rs7)
        feature_rows.append(feat_df)

    all_features = pd.concat(feature_rows, ignore_index=True)
    model = load_demand_model()
    predictions = model.predict(all_features)
    
    output_df["Predicted_Order_Demand"] = np.maximum(0.0, np.round(predictions, 2))
    
    if "rolling_std_7" in output_df.columns:
        output_df["Recommended_Safety_Stock"] = np.round(output_df["Predicted_Order_Demand"] + (1.65 * output_df["rolling_std_7"].astype(float))).astype(int)
    else:
        output_df["Recommended_Safety_Stock"] = np.round(output_df["Predicted_Order_Demand"] * 1.15).astype(int)

    return output_df


def generate_sample_batch_csv() -> pd.DataFrame:
    """Generate a ready-to-use sample CSV template for testing batch predictions."""
    sample_data = {
        "Product_Code": ["Product_0993", "Product_0979", "Product_1521", "Product_1507", "Product_1724"],
        "Warehouse": ["Whse_J", "Whse_J", "Whse_S", "Whse_C", "Whse_A"],
        "Product_Category": ["Category_028", "Category_028", "Category_019", "Category_019", "Category_003"],
        "Date": ["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02", "2026-09-03"],
        "lag_1": [520.0, 490.0, 1200.0, 310.0, 850.0],
        "lag_7": [500.0, 480.0, 1150.0, 300.0, 820.0],
        "lag_30": [480.0, 460.0, 1100.0, 290.0, 800.0],
        "rolling_mean_7": [510.0, 485.0, 1180.0, 305.0, 840.0],
        "rolling_std_7": [45.0, 40.0, 95.0, 25.0, 60.0]
    }
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    sample_df = generate_sample_batch_csv()
    results_df = predict_batch_demand(sample_df)
    print("Batch predictions successfully calculated:")
    print(results_df[["Product_Code", "Warehouse", "Predicted_Order_Demand", "Recommended_Safety_Stock"]])
