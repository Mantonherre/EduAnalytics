import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


FEATURES = ["asistencia", "participacion", "tareas_entregadas", "nota_examen"]
TARGET = "en_riesgo"


def entrenar_modelo(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() >= 2 else None
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X_train_s, y_train)
    return modelo, scaler, X_test_s, y_test


def calcular_metricas(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
    }
    try:
        metricas["auc_roc"] = round(roc_auc_score(y_test, y_prob), 3)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metricas["roc_curve"] = (fpr.tolist(), tpr.tolist())
    except ValueError:
        metricas["auc_roc"] = None
        metricas["roc_curve"] = None
    metricas["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metricas


def predecir_estudiante(modelo, scaler, datos: list) -> dict:
    X = scaler.transform([datos])
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]
    return {"prediccion": int(pred), "probabilidad_riesgo": round(float(prob), 3)}


def predecir_todos(modelo, scaler, df: pd.DataFrame) -> pd.DataFrame:
    X = scaler.transform(df[FEATURES].values)
    probs = modelo.predict_proba(X)[:, 1]
    result = df[["id", "nombre"]].copy()
    result["prob_riesgo"] = np.round(probs, 3)
    result["prediccion_riesgo"] = (probs >= 0.5).astype(int)
    return result.sort_values("prob_riesgo", ascending=False)
