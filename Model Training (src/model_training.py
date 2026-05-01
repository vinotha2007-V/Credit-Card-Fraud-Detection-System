from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    return model
