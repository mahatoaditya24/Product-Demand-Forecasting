import joblib
from pathlib import Path
from data_preprocessing import (
    load_invoice_data,
    apply_label,
    split_data,
    scale_features
)
from modeling_evaluation import train_random_forest, evaluate_classifier

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
     "total_item_dollars"
]

TARGET = "flag_invoice"

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    db_path = BASE_DIR / "data" / "inventory.db"
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    scaler_path = model_dir / "scaler.pkl"
    model_path = model_dir / "predict_flag_invoice.pkl"

    # Load data
    df = load_invoice_data(db_path)
    df = apply_label(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, scaler_path
    ) 

    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save the best model
    joblib.dump(grid_search.best_estimator_, model_path)


if __name__ == "__main__":
    main()


    
