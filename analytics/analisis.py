import pandas as pd
import numpy as np


def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta, encoding="utf-8")


def calcular_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    numericas = ["nota_media", "asistencia", "participacion", "tareas_entregadas", "nota_examen"]
    stats = df[numericas].describe().T
    stats["mediana"] = df[numericas].median()
    return stats.round(2)


def detectar_en_riesgo(df: pd.DataFrame, umbral_nota: float = 5.0, umbral_asistencia: float = 50.0) -> pd.DataFrame:
    mask = (df["nota_media"] < umbral_nota) | (df["asistencia"] < umbral_asistencia)
    return df[mask].copy()


def calcular_hash_registro(fila: pd.Series) -> str:
    import hashlib
    contenido = f"{fila['id']}|{fila['nota_media']}|{fila['asistencia']}|{fila['participacion']}|{fila['tareas_entregadas']}|{fila['nota_examen']}"
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def generar_recomendacion(fila: pd.Series) -> str:
    recomendaciones = []
    if fila["asistencia"] < 60:
        recomendaciones.append("Mejorar la asistencia a clase (actualmente por debajo del 60%).")
    if fila["participacion"] < 5:
        recomendaciones.append("Aumentar la participacion activa en el aula.")
    if fila["tareas_entregadas"] < 70:
        recomendaciones.append("Entregar el porcentaje minimo de tareas requeridas (>70%).")
    if fila["nota_media"] < 5:
        recomendaciones.append("Solicitar tutoria individualizada con el profesor.")
    if not recomendaciones:
        return "Buen rendimiento. Mantener el ritmo actual de trabajo."
    return " | ".join(recomendaciones)


def construir_bloques_desde_df(df: pd.DataFrame) -> list[dict]:
    bloques = []
    for _, fila in df.iterrows():
        bloques.append({
            "id_estudiante": fila["id"],
            "hash_registro": calcular_hash_registro(fila),
            "nota_media": float(fila["nota_media"]),
            "en_riesgo": int(fila["en_riesgo"])
        })
    return bloques
