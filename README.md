import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load Dataset
data = pd.read_csv("dataset/ev_data.csv")

# Features
X = data[['Speed','Acceleration','Battery','Temperature']]

# Target
y = data['Energy']

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

accuracy=r2_score(y_test,prediction)

print("Model Accuracy :",accuracy)

joblib.dump(model,"models/ev_model.pkl")
