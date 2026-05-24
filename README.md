# EduAnalytics Secure

Herramienta de Learning Analytics con Criptografía y Blockchain.  
Trabajo académico — Análisis de Datos II.

---

## Requisitos

- Python 3.10 o superior
- pip

---

## Instalación de dependencias

Desde la carpeta del proyecto:

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar la aplicación

### Paso 1 — Abrir una terminal

Abre PowerShell, CMD o la terminal de tu sistema operativo.

### Paso 2 — Situarse en la carpeta del proyecto

```bash
cd ruta/a/la/carpeta/EduAnalytics
```

### Paso 3 — Ejecutar la app

```bash
python -m streamlit run app.py
```

### Paso 4 — Abrir en el navegador

Una vez que el terminal muestre `You can now view your Streamlit app in your browser`, abre:

```
http://localhost:8501
```

### Paso 5 — Cerrar la app

Vuelve al terminal y pulsa `Ctrl + C`.

---

## Estructura del proyecto

```
EduAnalytics/
├── app.py                  ← App principal Streamlit
├── requirements.txt        ← Dependencias Python
├── clave.key               ← Clave AES (generada automáticamente)
├── data/
│   ├── generar_datos.py    ← Regenera el dataset sintético
│   └── estudiantes.csv     ← Dataset de 50 estudiantes
├── crypto/
│   └── cifrado_aes.py      ← Cifrado/descifrado AES (Fernet)
├── blockchain/
│   └── cadena.py           ← Simulación blockchain SHA-256
└── analytics/
    ├── analisis.py         ← Estadísticas y detección de riesgo
    └── prediccion.py       ← Modelo de predicción (regresión logística)
```

---

## Dependencias

```
streamlit · cryptography · pandas · scikit-learn · matplotlib · seaborn · numpy · python-docx
```
