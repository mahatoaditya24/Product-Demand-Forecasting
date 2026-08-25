import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "inventory.db"


def load_vendor_invoice_data(db_path=None):
    """
    Load vendor invoice data from SQLite database.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable.
    """
    X = df[['Quantity', 'Dollars']]
    y = df['Freight']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    