from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Invoice flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "Invoice flagging" / "models" / "scaler.pkl"

# Load once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_invoice_flag(input_data):
    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predict_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)