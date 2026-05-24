import os
from cryptography.fernet import Fernet

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clave.key")


def generar_clave() -> bytes:
    clave = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(clave)
    return clave


def cargar_clave() -> bytes:
    if not os.path.exists(KEY_FILE):
        return generar_clave()
    with open(KEY_FILE, "rb") as f:
        return f.read()


def cifrar(texto: str, clave: bytes) -> str:
    f = Fernet(clave)
    return f.encrypt(texto.encode("utf-8")).decode("utf-8")


def descifrar(texto_cifrado: str, clave: bytes) -> str:
    f = Fernet(clave)
    return f.decrypt(texto_cifrado.encode("utf-8")).decode("utf-8")


def cifrar_columnas(df, columnas: list, clave: bytes):
    df = df.copy()
    for col in columnas:
        df[col] = df[col].astype(str).apply(lambda x: cifrar(x, clave))
    return df


def descifrar_columnas(df, columnas: list, clave: bytes):
    df = df.copy()
    for col in columnas:
        df[col] = df[col].apply(lambda x: descifrar(x, clave))
    return df
