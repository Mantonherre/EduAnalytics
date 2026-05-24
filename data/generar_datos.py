import pandas as pd
import numpy as np
import os

np.random.seed(42)

NOMBRES = [
    "Alejandro García", "María López", "Carlos Martínez", "Ana Rodríguez", "Luis Hernández",
    "Laura González", "Miguel Pérez", "Elena Sánchez", "Pablo Díaz", "Carmen Moreno",
    "Javier Torres", "Isabel Romero", "Adrián Flores", "Sofía Jiménez", "Daniel Ruiz",
    "Lucía Navarro", "Sergio Morales", "Marta Ortega", "Rubén Castro", "Clara Vargas",
    "Tomás Ramos", "Patricia Herrera", "Andrés Medina", "Valeria Suárez", "Óscar Guerrero",
    "Natalia Reyes", "Eduardo Vega", "Cristina Blanco", "Marco Mendoza", "Paula Domínguez",
    "Alberto Silva", "Beatriz Cortés", "Rodrigo Aguilar", "Nuria Molina", "Fernando Cabrera",
    "Sandra Fuentes", "Guillermo Bravo", "Alicia Delgado", "Ignacio Rojas", "Raquel Soto",
    "Enrique Espinoza", "Lorena Cárdenas", "Héctor Reina", "Verónica Leal", "Antonio Moya",
    "Rosa Vidal", "Iván Ponce", "Pilar Montoya", "Ernesto Salas", "Inés Caballero"
]

CORREOS = [f"{nombre.split()[0].lower()}.{nombre.split()[1].lower()}@edu.es" for nombre in NOMBRES]

n = 50
asistencia = np.clip(np.random.normal(78, 18, n), 10, 100)
participacion = np.clip(np.random.normal(6.5, 2.2, n), 0, 10)
tareas = np.clip(np.random.normal(75, 20, n), 0, 100)
nota_media = np.clip(
    0.3 * (asistencia / 10) + 0.3 * participacion + 0.3 * (tareas / 10) + np.random.normal(0, 0.5, n),
    0, 10
)
nota_examen = np.clip(nota_media * 0.85 + np.random.normal(0, 0.8, n), 0, 10)
en_riesgo = ((nota_media < 5.0) | (asistencia < 50)).astype(int)

df = pd.DataFrame({
    "id": [f"EST{str(i+1).zfill(3)}" for i in range(n)],
    "nombre": NOMBRES,
    "correo": CORREOS,
    "nota_media": np.round(nota_media, 2),
    "asistencia": np.round(asistencia, 1),
    "participacion": np.round(participacion, 2),
    "tareas_entregadas": np.round(tareas, 1),
    "nota_examen": np.round(nota_examen, 2),
    "en_riesgo": en_riesgo
})

out = os.path.join(os.path.dirname(__file__), "estudiantes.csv")
df.to_csv(out, index=False, encoding="utf-8")
print(f"Dataset generado: {out} ({len(df)} estudiantes)")
print(f"Estudiantes en riesgo: {df['en_riesgo'].sum()} ({df['en_riesgo'].mean()*100:.1f}%)")
