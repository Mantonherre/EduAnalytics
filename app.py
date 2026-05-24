import os
import sys
import hashlib
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.cifrado_aes import cargar_clave, generar_clave, cifrar, descifrar, cifrar_columnas, descifrar_columnas
from blockchain.cadena import CadenaEducativa
from analytics.analisis import (
    cargar_datos, calcular_estadisticas, detectar_en_riesgo,
    generar_recomendacion, construir_bloques_desde_df
)
from analytics.prediccion import entrenar_modelo, calcular_metricas, predecir_todos

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduAnalytics Secure",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "estudiantes.csv")

# ── Estado de sesion ─────────────────────────────────────────────────────────
if "clave" not in st.session_state:
    st.session_state.clave = cargar_clave()
if "cadena" not in st.session_state:
    st.session_state.cadena = CadenaEducativa()
    st.session_state.cadena_iniciada = False
if "df_original" not in st.session_state:
    st.session_state.df_original = cargar_datos(DATA_PATH)
if "modelo" not in st.session_state:
    st.session_state.modelo = None
    st.session_state.scaler = None
    st.session_state.metricas = None


def color_riesgo(val):
    color = "#ffcccc" if val == 1 else "#ccffcc"
    return f"background-color: {color}"


# ── Función auxiliar para explicar cada apartado al profesor ─────────────────
def explicacion_profesor(titulo, texto):
    """
    Muestra una explicación desplegable dentro de la app.

    La idea es que el profesor pueda ir leyendo, paso a paso, qué hace cada
    sección sin tener que interpretar directamente el código fuente. Uso un
    expander para no saturar la interfaz: la explicación está disponible, pero
    solo se despliega cuando se quiere leer.
    """
    with st.expander(f"📘 Explicación para el profesor: {titulo}", expanded=False):
        st.markdown(texto)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
    st.title("EduAnalytics Secure")
    st.caption("Herramienta de Learning Analytics con Criptografía y Blockchain")
    st.divider()
    pagina = st.radio(
        "Navegación",
        ["Inicio", "Datos y Cifrado", "Criptografía AES", "Blockchain", "Analytics y ML"],
        index=0
    )
    st.divider()
    st.caption("Python 3.13 · Streamlit · Fernet/AES · SHA-256")

# ════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — INICIO
# ════════════════════════════════════════════════════════════════════════════
if pagina == "Inicio":
    st.title("🎓 EduAnalytics Secure")
    st.subheader("Herramienta de Learning Analytics con Criptografía y Blockchain")
    explicacion_profesor(
        "visión general del proyecto",
        """
        En esta pantalla se presenta el objetivo global del proyecto. La aplicación
        combina tres partes: protección de datos personales mediante cifrado,
        verificación de integridad mediante blockchain y análisis educativo mediante
        técnicas de analítica y aprendizaje automático.

        Esta introducción sirve para que el profesor entienda que no es una app
        aislada, sino una demostración integrada de seguridad, privacidad y toma de
        decisiones educativas basada en datos.
        """
    )
    st.markdown("---")

    explicacion_profesor(
        "botones de la simulación",
        """
        Los tres botones representan el ciclo completo de la demostración:

        1. **Construir cadena**: transforma los registros del dataset en bloques.
        2. **Verificar integridad**: comprueba que los hashes siguen coincidiendo.
        3. **Simular manipulación**: modifica un bloque a propósito para comprobar
           que la verificación detecta el cambio.
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🔐 Criptografía AES\nCifrado simétrico Fernet (AES-128-CBC) para proteger datos sensibles de estudiantes como nombres y correos electrónicos.")
    with col2:
        st.warning("### ⛓️ Blockchain\nCadena de bloques con SHA-256 para garantizar la integridad e inmutabilidad de los registros académicos. Cualquier modificación se detecta automáticamente.")
    with col3:
        st.success("### 📊 Learning Analytics\nAnálisis estadístico, visualizaciones y modelo de predicción de riesgo académico basado en regresión logística.")

    st.markdown("---")
    st.markdown("""
    ### Contexto y Objetivos

    Las plataformas de aprendizaje online gestionan una cantidad creciente de datos educativos sensibles:
    notas, asistencia, participación y comportamiento del alumnado. La ausencia de mecanismos adecuados
    de protección puede comprometer tanto la **privacidad** de los estudiantes como la **integridad** de
    los registros académicos.

    Esta herramienta demuestra cómo combinar tres tecnologías clave para abordar este problema:

    | Componente | Tecnología | Propósito |
    |-----------|-----------|-----------|
    | Cifrado de datos | AES/Fernet (cryptography) | Proteger información personal |
    | Integridad de registros | SHA-256 / Blockchain | Detectar manipulaciones |
    | Análisis de rendimiento | Pandas + Scikit-learn | Identificar estudiantes en riesgo |

    ### Público Objetivo
    - **Profesores**: Acceso a análisis de rendimiento con datos anonimizados
    - **Administradores**: Verificación de integridad de actas y expedientes
    - **Investigadores**: Estudio de patrones de aprendizaje con garantías éticas
    """)

    st.markdown("---")
    df = st.session_state.df_original
    explicacion_profesor(
        "métricas iniciales",
        """
        Aquí se cargan los datos originales de estudiantes que se guardaron en
        `st.session_state`. De esta forma, Streamlit conserva los datos mientras el
        usuario navega por la aplicación sin tener que leer el CSV continuamente.

        Las métricas resumen muestran una primera lectura rápida del conjunto de
        datos: número total de estudiantes, cuántos están marcados como en riesgo,
        nota media global y asistencia media.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estudiantes", len(df))
    col2.metric("En riesgo", int(df["en_riesgo"].sum()))
    col3.metric("Nota media", f"{df['nota_media'].mean():.2f}")
    col4.metric("Asistencia media", f"{df['asistencia'].mean():.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — DATOS Y CIFRADO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "Datos y Cifrado":
    st.title("📋 Datos de Estudiantes y Cifrado AES")
    explicacion_profesor(
        "carga y copia del dataset",
        """
        En esta página se trabaja con una copia del dataset original. Hacer
        `copy()` evita modificar accidentalmente los datos originales guardados en
        memoria. Esto es importante porque algunas operaciones, como cifrar columnas
        o añadir nuevas columnas auxiliares, solo deben afectar a la visualización.
        """
    )
    df = st.session_state.df_original.copy()

    mostrar_cifrado = st.toggle("Mostrar nombres y correos cifrados", value=False)
    umbral = st.slider("Umbral de riesgo (nota)", 3.0, 7.0, 5.0, 0.5)

    st.subheader("⚠️ Estudiantes en Riesgo Académico")
    explicacion_profesor(
        "detección por umbral",
        """
        En esta parte se aplica una regla sencilla: se considera estudiante en riesgo
        aquel cuya nota media está por debajo del umbral seleccionado. El profesor
        puede mover el deslizador para comprobar cómo cambia el número de estudiantes
        detectados.

        Esta detección por umbral es interpretable y fácil de justificar, por eso se
        muestra antes del modelo predictivo más avanzado.
        """
    )
    en_riesgo = detectar_en_riesgo(df, umbral_nota=umbral)
    if len(en_riesgo) > 0:
        st.error(f"Se han detectado **{len(en_riesgo)} estudiantes** en situación de riesgo académico.")
        en_riesgo["recomendacion"] = en_riesgo.apply(generar_recomendacion, axis=1)
        st.dataframe(
            en_riesgo[["id", "nombre", "nota_media", "asistencia", "tareas_entregadas", "recomendacion"]].rename(
                columns={"id": "ID", "nombre": "Nombre", "nota_media": "Nota media",
                         "asistencia": "Asistencia (%)", "tareas_entregadas": "Tareas (%)",
                         "recomendacion": "Recomendación"}
            ),
            width='stretch'
        )
    else:
        st.success("Ningún estudiante en riesgo con el umbral actual.")

    st.markdown("---")
    if mostrar_cifrado:
        clave = st.session_state.clave
        df_mostrar = cifrar_columnas(df, ["nombre", "correo"], clave)
        st.info("🔐 Los campos **nombre** y **correo** se muestran cifrados con AES-128 (Fernet). Solo quien posea la clave puede descifrarlos.")
        explicacion_profesor(
            "cifrado de columnas sensibles",
            """
            Cuando el interruptor está activado, la app no muestra directamente el
            nombre ni el correo del estudiante. En su lugar, aplica la función
            `cifrar_columnas()` a esas dos columnas.

            Esto demuestra una práctica habitual en protección de datos: separar la
            información académica útil para el análisis de los identificadores
            personales. Así, un profesor o investigador puede analizar rendimiento
            sin exponer datos personales innecesariamente.
            """
        )
    else:
        df_mostrar = df.copy()
        explicacion_profesor(
            "visualización sin cifrado",
            """
            Si el interruptor está desactivado, la tabla se muestra con los datos
            originales para facilitar la lectura durante la demostración. Esto permite
            comparar de forma clara la diferencia entre trabajar con datos visibles y
            trabajar con datos protegidos.
            """
        )

    df_mostrar["en_riesgo_texto"] = df_mostrar["en_riesgo"].map({0: "No", 1: "⚠️ Sí"})

    st.dataframe(
        df_mostrar[["id", "nombre", "correo", "nota_media", "asistencia",
                    "participacion", "tareas_entregadas", "nota_examen", "en_riesgo_texto"]].rename(
            columns={
                "id": "ID", "nombre": "Nombre", "correo": "Correo",
                "nota_media": "Nota media", "asistencia": "Asistencia (%)",
                "participacion": "Participación", "tareas_entregadas": "Tareas (%)",
                "nota_examen": "Nota examen", "en_riesgo_texto": "En riesgo"
            }
        ),
        width='stretch',
        height=400
    )

    st.markdown("---")
    st.subheader("Estadísticas descriptivas")
    explicacion_profesor(
        "estadísticas descriptivas",
        """
        Las estadísticas descriptivas resumen el comportamiento general del grupo:
        medias, desviaciones, mínimos y máximos. Antes de aplicar modelos de machine
        learning conviene observar estos valores porque permiten detectar patrones,
        datos extremos o variables que pueden estar relacionadas con el riesgo
        académico.
        """
    )
    stats = calcular_estadisticas(df)
    st.dataframe(stats, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — CRIPTOGRAFÍA AES
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "Criptografía AES":
    st.title("🔐 Criptografía AES — Demostración Interactiva")
    explicacion_profesor(
        "objetivo de la demostración AES",
        """
        Esta sección permite explicar de forma práctica cómo se protege un dato
        sensible. En lugar de limitarse a decir que se usa cifrado, la app permite
        escribir un texto, cifrarlo, ver el resultado ilegible y después descifrarlo
        usando la misma clave.
        """
    )

    st.markdown("""
    **AES (Advanced Encryption Standard)** es el estándar de cifrado simétrico más utilizado a nivel mundial.
    Esta herramienta utiliza **Fernet**, una implementación de AES-128-CBC con HMAC-SHA256 para autenticación,
    disponible en la librería `cryptography` de Python.

    El cifrado simétrico usa la **misma clave** para cifrar y descifrar, lo que lo hace ideal para
    proteger grandes volúmenes de datos como los registros de estudiantes.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gestión de Claves")
        clave_actual = st.session_state.clave
        st.code(f"Clave actual (base64):\n{clave_actual.decode('utf-8')}", language="text")

        explicacion_profesor(
            "gestión de la clave",
            """
            La clave es el elemento central del cifrado simétrico. Si se pierde, no
            se pueden recuperar los datos cifrados; si se comparte indebidamente,
            cualquiera podría descifrarlos. Por eso la app muestra la clave actual y
            permite generar una nueva para demostrar la dependencia entre clave,
            cifrado y descifrado.
            """
        )

        if st.button("🔄 Generar nueva clave AES"):
            st.session_state.clave = generar_clave()
            st.success("Nueva clave generada y guardada en clave.key")
            st.rerun()

        st.info("⚠️ Si cambias la clave, los datos previamente cifrados ya no podrán descifrarse con la clave nueva.")

    with col2:
        st.subheader("Cifrar y Descifrar Texto")
        explicacion_profesor(
            "flujo de cifrado y descifrado",
            """
            El usuario introduce un texto en claro, por ejemplo un nombre de
            estudiante. Al pulsar **Cifrar**, el texto se transforma en una cadena
            ilegible. Al pulsar **Descifrar**, la app usa la misma clave para recuperar
            el texto original.

            Esto demuestra que AES/Fernet es reversible únicamente si se dispone de
            la clave correcta.
            """
        )
        texto_entrada = st.text_area("Texto a cifrar:", value="Juan García Martínez", height=80)

        col_a, col_b = st.columns(2)
        cifrado_resultado = ""
        with col_a:
            if st.button("🔒 Cifrar", width='stretch'):
                try:
                    cifrado_resultado = cifrar(texto_entrada, st.session_state.clave)
                    st.session_state.ultimo_cifrado = cifrado_resultado
                    st.success("Texto cifrado correctamente")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_b:
            if st.button("🔓 Descifrar último cifrado", width='stretch'):
                try:
                    ultimo = st.session_state.get("ultimo_cifrado", "")
                    if ultimo:
                        desc = descifrar(ultimo, st.session_state.clave)
                        st.success(f"Texto descifrado: **{desc}**")
                    else:
                        st.warning("No hay texto cifrado en memoria. Cifra algo primero.")
                except Exception as e:
                    st.error(f"Error al descifrar: {e}")

        if "ultimo_cifrado" in st.session_state and st.session_state.ultimo_cifrado:
            st.code(f"Texto cifrado:\n{st.session_state.ultimo_cifrado}", language="text")

    st.markdown("---")
    st.subheader("¿Cómo Funciona Fernet/AES?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Generación de clave**
        ```python
        from cryptography.fernet import Fernet
        clave = Fernet.generate_key()
        # → clave aleatoria 256 bits
        ```
        """)
    with col2:
        st.markdown("""
        **2. Cifrado**
        ```python
        f = Fernet(clave)
        cifrado = f.encrypt(b"datos")
        # → bytes cifrados + IV + HMAC
        ```
        """)
    with col3:
        st.markdown("""
        **3. Descifrado**
        ```python
        f = Fernet(clave)
        original = f.decrypt(cifrado)
        # → datos originales verificados
        ```
        """)

    st.markdown("---")
    st.subheader("Comparación de Algoritmos")
    comparacion = pd.DataFrame({
        "Algoritmo": ["AES-128 (Fernet)", "RSA-2048", "SHA-256"],
        "Tipo": ["Simétrico", "Asimétrico", "Hash (no reversible)"],
        "Uso": ["Cifrar datos masivos", "Firmas digitales / intercambio de clave", "Integridad / Blockchain"],
        "Velocidad": ["Muy alta", "Baja", "Muy alta"],
        "Reversible": ["Sí (con clave)", "Sí (con clave privada)", "No"]
    })
    st.dataframe(comparacion, width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — BLOCKCHAIN
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "Blockchain":
    st.title("⛓️ Blockchain — Integridad de Registros Educativos")
    explicacion_profesor(
        "objetivo de la blockchain",
        """
        Esta página no usa blockchain para criptomonedas, sino como mecanismo de
        integridad. El objetivo es demostrar que, si un registro académico se modifica
        después de haber sido guardado en la cadena, el sistema puede detectar esa
        alteración.
        """
    )

    st.markdown("""
    La simulación blockchain de esta herramienta usa **SHA-256** para encadenar bloques de registros educativos.
    Cada bloque contiene el hash del bloque anterior, formando una cadena donde cualquier modificación
    en un registro quiebra la cadena y es detectada automáticamente.
    """)

    df = st.session_state.df_original
    cadena = st.session_state.cadena

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏗️ Construir cadena desde dataset", width='stretch'):
            cadena = CadenaEducativa()
            bloques_datos = construir_bloques_desde_df(df)
            for lote in [bloques_datos[i:i+10] for i in range(0, len(bloques_datos), 10)]:
                cadena.agregar_bloque({"registros": lote, "total": len(lote)})
            st.session_state.cadena = cadena
            st.session_state.cadena_iniciada = True
            st.success(f"Cadena construida con {len(cadena.cadena)} bloques (1 génesis + {len(cadena.cadena)-1} de datos)")

    with col2:
        if st.button("✅ Verificar integridad", width='stretch'):
            valida, errores = cadena.verificar_integridad()
            if valida:
                st.success("✅ Cadena íntegra. Ningún registro ha sido manipulado.")
            else:
                st.error(f"❌ Cadena corrupta. {len(errores)} error(es) detectado(s):")
                for e in errores:
                    st.error(f"  • {e}")

    with col3:
        if st.button("⚠️ Simular manipulación (bloque 1)", width='stretch'):
            if len(cadena.cadena) > 1:
                cadena.simular_manipulacion(1, {"manipulado": True, "nota_fraudulenta": 10.0})
                st.session_state.cadena = cadena
                st.warning("Se ha modificado el bloque 1 directamente. Verifica la integridad para ver el resultado.")
                st.info("📘 Explicación: al cambiar los datos de un bloque, su hash ya no coincide con el que espera la cadena. Por eso la verificación posterior debe fallar.")
            else:
                st.warning("Construye la cadena primero.")

    st.markdown("---")
    st.subheader("Visualización de la Cadena")
    explicacion_profesor(
        "visualización de bloques",
        """
        Cada bloque muestra dos hashes: el hash propio y el hash del bloque anterior.
        Esta relación es lo que encadena los bloques. Si un bloque intermedio cambia,
        su hash cambia también y deja de coincidir con el valor almacenado en el
        bloque siguiente.
        """
    )

    if len(cadena.cadena) > 0:
        lista_bloques = cadena.to_lista()
        for bloque in lista_bloques[:6]:
            with st.expander(f"Bloque {bloque['indice']} — {bloque['timestamp']}", expanded=(bloque['indice'] == 0)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Hash anterior:** `{bloque['hash_anterior']}`")
                    st.markdown(f"**Hash propio:** `{bloque['hash']}`")
                with col_b:
                    st.json(bloque["datos"] if isinstance(bloque["datos"], dict) and len(str(bloque["datos"])) < 300
                            else {"resumen": f"{len(bloque['datos'].get('registros', []))} registros" if "registros" in bloque.get("datos", {}) else "..."})

        if len(lista_bloques) > 6:
            st.info(f"... y {len(lista_bloques) - 6} bloques más. Total: {len(lista_bloques)} bloques en la cadena.")

    st.markdown("---")
    st.subheader("¿Por qué SHA-256 para Blockchain?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Propiedades de SHA-256:**
        - **Determinista**: el mismo input siempre produce el mismo hash
        - **Efecto avalancha**: un cambio mínimo produce un hash completamente diferente
        - **Irreversible**: no es posible obtener el input a partir del hash
        - **Resistente a colisiones**: prácticamente imposible que dos inputs distintos produzcan el mismo hash
        """)
    with col2:
        ejemplo = "EST001|7.45|85.0|6.5|90.0|6.82"
        hash_ejemplo = hashlib.sha256(ejemplo.encode()).hexdigest()
        ejemplo_mod = "EST001|7.46|85.0|6.5|90.0|6.82"
        hash_mod = hashlib.sha256(ejemplo_mod.encode()).hexdigest()
        st.markdown("**Demostración del efecto avalancha:**")
        st.code(f"Input original:  {ejemplo}\nSHA-256:         {hash_ejemplo}", language="text")
        st.code(f"Input +0.01:     {ejemplo_mod}\nSHA-256:         {hash_mod}", language="text")
        st.caption("Cambiar 7.45 → 7.46 produce un hash completamente diferente.")


# ════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — ANALYTICS Y ML
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "Analytics y ML":
    st.title("📊 Learning Analytics y Predicción de Riesgo")
    explicacion_profesor(
        "objetivo de analytics y machine learning",
        """
        Esta sección transforma los datos académicos en información útil para la
        toma de decisiones. Primero se muestran visualizaciones para entender los
        patrones del dataset. Después se entrena un modelo predictivo para estimar
        riesgo académico. Finalmente se generan recomendaciones que sirven como apoyo
        al criterio docente.
        """
    )

    df = st.session_state.df_original

    tab1, tab2, tab3 = st.tabs(["📈 Visualizaciones", "🤖 Modelo Predictivo", "📋 Recomendaciones"])

    # ── Tab 1: Visualizaciones ────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución de Notas Medias")
            st.caption("Este histograma compara la distribución de notas entre estudiantes en riesgo y sin riesgo. La línea discontinua marca el umbral de aprobado usado como referencia.")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[df["en_riesgo"] == 0]["nota_media"], bins=15, alpha=0.7,
                    color="#2196F3", label="Sin riesgo", edgecolor="white")
            ax.hist(df[df["en_riesgo"] == 1]["nota_media"], bins=8, alpha=0.8,
                    color="#F44336", label="En riesgo", edgecolor="white")
            ax.axvline(x=5.0, color="orange", linestyle="--", linewidth=2, label="Umbral (5.0)")
            ax.set_xlabel("Nota media")
            ax.set_ylabel("Número de estudiantes")
            ax.legend()
            ax.set_facecolor("#f8f9fa")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Asistencia vs. Nota Media")
            st.caption("Cada punto representa un estudiante. El gráfico permite observar si la asistencia se relaciona con la nota media y con la etiqueta de riesgo.")
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = df["en_riesgo"].map({0: "#2196F3", 1: "#F44336"})
            ax.scatter(df["asistencia"], df["nota_media"], c=colors, alpha=0.7, s=60, edgecolors="white")
            ax.axhline(y=5.0, color="orange", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.axvline(x=50, color="orange", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Asistencia (%)")
            ax.set_ylabel("Nota media")
            sin_riesgo = mpatches.Patch(color="#2196F3", label="Sin riesgo")
            en_riesgo_patch = mpatches.Patch(color="#F44336", label="En riesgo")
            ax.legend(handles=[sin_riesgo, en_riesgo_patch])
            ax.set_facecolor("#f8f9fa")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Mapa de Calor — Correlaciones")
            st.caption("El mapa de calor muestra qué variables se mueven juntas. Valores cercanos a 1 indican relación positiva; valores cercanos a -1 indican relación negativa.")
            fig, ax = plt.subplots(figsize=(6, 5))
            numericas = ["nota_media", "asistencia", "participacion", "tareas_entregadas", "nota_examen", "en_riesgo"]
            corr = df[numericas].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                        center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col4:
            st.subheader("Nivel de Participación por Grupo de Riesgo")
            st.caption("Este gráfico compara la participación media de los estudiantes en riesgo frente a los que no están en riesgo. Las barras de error muestran la variabilidad.")
            fig, ax = plt.subplots(figsize=(6, 4))
            grupos = ["Sin riesgo", "En riesgo"]
            medias = [df[df["en_riesgo"] == 0]["participacion"].mean(),
                      df[df["en_riesgo"] == 1]["participacion"].mean()]
            errores = [df[df["en_riesgo"] == 0]["participacion"].std(),
                       df[df["en_riesgo"] == 1]["participacion"].std()]
            bars = ax.bar(grupos, medias, color=["#2196F3", "#F44336"],
                          yerr=errores, capsize=5, alpha=0.8, edgecolor="white")
            ax.set_ylabel("Participación media (0-10)")
            ax.set_ylim(0, 10)
            for bar, val in zip(bars, medias):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
            ax.set_facecolor("#f8f9fa")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 2: Modelo Predictivo ──────────────────────────────────────────
    with tab2:
        st.subheader("Modelo de Predicción de Riesgo Académico")
        explicacion_profesor(
            "entrenamiento del modelo",
            """
            El modelo se entrena con datos históricos para aprender patrones asociados
            al riesgo académico. Las variables de entrada son asistencia, participación,
            tareas entregadas y nota de examen. La salida esperada es si el estudiante
            está o no en riesgo.

            La regresión logística es adecuada para una demostración educativa porque
            no funciona como una caja negra total: permite explicar que calcula una
            probabilidad de pertenecer a la clase "en riesgo".
            """
        )
        st.markdown("""
        El modelo usa **regresión logística** entrenada sobre las variables: asistencia,
        participación, porcentaje de tareas entregadas y nota del examen.
        La elección de regresión logística responde a su **interpretabilidad**: permite
        conocer qué variables tienen más peso en la predicción.
        """)

        if st.button("🚀 Entrenar modelo", width='content'):
            with st.spinner("Entrenando..."):
                modelo, scaler, X_test, y_test = entrenar_modelo(df)
                metricas = calcular_metricas(modelo, X_test, y_test)
                st.session_state.modelo = modelo
                st.session_state.scaler = scaler
                st.session_state.metricas = metricas
                st.success("Modelo entrenado correctamente.")
                st.info("📘 Explicación: el dataset se divide internamente en entrenamiento y prueba. El modelo aprende con una parte de los datos y se evalúa con otra parte para medir si generaliza.")

        if st.session_state.metricas:
            metricas = st.session_state.metricas
            explicacion_profesor(
                "métricas de evaluación",
                """
                Las métricas permiten valorar la calidad del modelo:

                - **Accuracy**: porcentaje total de aciertos.
                - **Recall**: capacidad para detectar correctamente estudiantes en riesgo.
                - **F1 Score**: equilibrio entre precisión y recall.
                - **AUC-ROC**: capacidad general para separar estudiantes en riesgo y sin riesgo.

                En este contexto educativo, el recall es especialmente importante
                porque interesa no dejar sin detectar a estudiantes que necesitan apoyo.
                """
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", metricas["accuracy"])
            col2.metric("Recall", metricas["recall"])
            col3.metric("F1 Score", metricas["f1"])
            col4.metric("AUC-ROC", metricas.get("auc_roc", "N/A"))

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Matriz de Confusión**")
                cm = metricas["confusion_matrix"]
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Sin riesgo", "En riesgo"],
                            yticklabels=["Sin riesgo", "En riesgo"])
                ax.set_xlabel("Predicción")
                ax.set_ylabel("Real")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_b:
                if metricas.get("roc_curve"):
                    st.markdown("**Curva ROC**")
                    fpr, tpr = metricas["roc_curve"]
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.plot(fpr, tpr, color="#2196F3", lw=2,
                            label=f"AUC = {metricas['auc_roc']:.3f}")
                    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
                    ax.set_xlabel("Tasa de Falsos Positivos")
                    ax.set_ylabel("Tasa de Verdaderos Positivos")
                    ax.legend()
                    ax.set_facecolor("#f8f9fa")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            st.markdown("---")
            st.subheader("Predicciones sobre el dataset completo")
            explicacion_profesor(
                "predicciones finales",
                """
                Una vez entrenado el modelo, se aplica a todos los estudiantes para
                obtener una probabilidad de riesgo. Esta probabilidad no debe entenderse
                como una sentencia automática, sino como una señal de alerta para que el
                profesor revise el caso con más información contextual.
                """
            )
            predicciones = predecir_todos(st.session_state.modelo, st.session_state.scaler, df)
            predicciones["prob_riesgo_pct"] = (predicciones["prob_riesgo"] * 100).round(1).astype(str) + "%"
            st.dataframe(
                predicciones[["id", "nombre", "prob_riesgo_pct", "prediccion_riesgo"]].rename(
                    columns={"id": "ID", "nombre": "Nombre",
                             "prob_riesgo_pct": "Probabilidad de riesgo",
                             "prediccion_riesgo": "Predicción (0=seguro, 1=riesgo)"}
                ),
                width='stretch'
            )

    # ── Tab 3: Recomendaciones ────────────────────────────────────────────
    with tab3:
        st.subheader("Recomendaciones Personalizadas")
        explicacion_profesor(
            "generación de recomendaciones",
            """
            Las recomendaciones se generan a partir de reglas interpretables sobre las
            métricas individuales. Por ejemplo, una asistencia baja puede generar una
            recomendación distinta a una baja entrega de tareas.

            Esto permite que la app no solo detecte riesgo, sino que sugiera posibles
            líneas de intervención docente.
            """
        )
        st.markdown("""
        Basadas en el análisis de las métricas individuales de cada estudiante, el sistema genera
        recomendaciones de intervención. Es fundamental recordar que estas sugerencias son un
        **apoyo al criterio docente**, no decisiones automáticas.
        """)

        df_rec = df.copy()
        df_rec["recomendacion"] = df_rec.apply(generar_recomendacion, axis=1)
        df_rec["estado"] = df_rec["en_riesgo"].map({0: "✅ Normal", 1: "⚠️ En riesgo"})

        filtro = st.selectbox("Mostrar:", ["Todos los estudiantes", "Solo en riesgo", "Solo sin riesgo"])
        if filtro == "Solo en riesgo":
            df_mostrar = df_rec[df_rec["en_riesgo"] == 1]
        elif filtro == "Solo sin riesgo":
            df_mostrar = df_rec[df_rec["en_riesgo"] == 0]
        else:
            df_mostrar = df_rec

        st.dataframe(
            df_mostrar[["id", "nombre", "nota_media", "asistencia", "estado", "recomendacion"]].rename(
                columns={"id": "ID", "nombre": "Nombre", "nota_media": "Nota media",
                         "asistencia": "Asistencia (%)", "estado": "Estado",
                         "recomendacion": "Recomendación"}
            ),
            width='stretch'
        )
