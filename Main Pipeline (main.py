from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import handle_imbalance
from src.model_training import train_model
from src.evaluate import evaluate_model
import joblib

df = load_data("data/creditcard.csv")

X, y = preprocess_data(df)

X_res, y_res = handle_imbalance(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)

joblib.dump(model, "models/fraud_model.pkl")
