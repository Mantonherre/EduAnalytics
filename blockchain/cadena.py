import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class Bloque:
    indice: int
    timestamp: float
    datos: dict
    hash_anterior: str
    hash: str = field(default="", init=False)

    def __post_init__(self):
        self.hash = self.calcular_hash()

    def calcular_hash(self) -> str:
        contenido = json.dumps({
            "indice": self.indice,
            "timestamp": self.timestamp,
            "datos": self.datos,
            "hash_anterior": self.hash_anterior
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(contenido.encode("utf-8")).hexdigest()

    def es_valido(self) -> bool:
        return self.hash == self.calcular_hash()


class CadenaEducativa:
    def __init__(self):
        self.cadena: list[Bloque] = []
        self._crear_bloque_genesis()

    def _crear_bloque_genesis(self):
        genesis = Bloque(
            indice=0,
            timestamp=time.time(),
            datos={"mensaje": "Bloque génesis — Inicio de la cadena de registros educativos"},
            hash_anterior="0" * 64
        )
        self.cadena.append(genesis)

    def agregar_bloque(self, datos: dict) -> Bloque:
        ultimo = self.cadena[-1]
        nuevo = Bloque(
            indice=len(self.cadena),
            timestamp=time.time(),
            datos=datos,
            hash_anterior=ultimo.hash
        )
        self.cadena.append(nuevo)
        return nuevo

    def verificar_integridad(self) -> tuple[bool, list[str]]:
        errores = []
        for i in range(1, len(self.cadena)):
            actual = self.cadena[i]
            anterior = self.cadena[i - 1]
            if not actual.es_valido():
                errores.append(f"Bloque {actual.indice}: hash interno corrupto")
            if actual.hash_anterior != anterior.hash:
                errores.append(f"Bloque {actual.indice}: enlace roto con bloque {anterior.indice}")
        return len(errores) == 0, errores

    def simular_manipulacion(self, indice: int, nuevos_datos: dict):
        if 0 < indice < len(self.cadena):
            self.cadena[indice].datos = nuevos_datos

    def to_lista(self) -> list[dict]:
        return [
            {
                "indice": b.indice,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b.timestamp)),
                "datos": b.datos,
                "hash_anterior": b.hash_anterior[:16] + "...",
                "hash": b.hash[:16] + "..."
            }
            for b in self.cadena
        ]
