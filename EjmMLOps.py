#pip install mlflow dvc git

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar el dataset Iris
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Iniciar el experimento de MLflow
with mlflow.start_run():
    print("Experimento iniciado...")

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print(f"Exactitud del modelo: {accuracy}")

mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "random_forest_model")

