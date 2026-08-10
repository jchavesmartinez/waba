"""
Motor de transformacion: raw -> tabla derivada consumible.

    raw_<cliente>.<tabla_origen>
        |  filtro        que filas aplican (SQL declarado en la metadata)
        |  extraccion    texto -> columnas (modelo/extractores.py)
        |  tipado        texto -> valor tipado (modelo/tipos.py)
        |  clasificacion override > mapeo > regla > sin_clasificar
        v
    semantic_<cliente>.<tabla_destino>   +  <tabla_destino>__rechazos

LA PROPIEDAD QUE SOSTIENE TODO ESTE MODULO es que la tabla derivada es una
FUNCION PURA de sus entradas:

    derivada = f(raw, mapeo, overrides, metadata)

De ahi sale todo lo demas. Reconstruir es DROP + CREATE, sin UPSERT, sin estado
incremental y sin migraciones: cambiar una regla y volver a correr basta para
corregir todo el historico. Es tambien lo que hace que "reconstruir completo" no
sea caro: el LLM NO participa de esta ruta. El modelo de lenguaje llena la tabla
de mapeo en un job aparte y esa tabla es una ENTRADA mas; reconstruir 100.000
filas son cero llamadas al LLM.

Si algun dia hace falta guardar algo que no viene de ninguna entrada —una
correccion escrita a mano directo en la tabla derivada— esta propiedad se pierde
y con ella todo lo anterior. La salida correcta en ese caso es declarar ese dato
como una entrada mas (una pestania de overrides que se lee), nunca editar la
tabla derivada.

RECHAZAR NO ES DESCARTAR. Una fila que pasa el filtro pero no produce sus campos
requeridos va a <tabla>__rechazos con el motivo. Se ve identico a "no habia
nada que extraer" y no lo es: puede ser el cuerpo truncado a 2000 caracteres,
un cambio de formato del banco o un reenvio con una cadena larga arriba. Los
tres son problemas que hay que ver, no filas que deban desaparecer en silencio.
"""

import hashlib
import json
import logging

from . import extractores, tipos
from .metadata import campos_de, clasificacion_de, overrides_de

logger = logging.getLogger("fachavi.modelo.motor")

# Columnas tecnicas que lleva toda tabla derivada.
COL_CLAVE = "_clave"
COL_ORIGEN = "_origen"
COL_MODELO = "_modelo_id"

_TIPOS_TECNICOS = {
    COL_CLAVE: "texto",
    COL_ORIGEN: "texto",
    COL_MODELO: "texto",
}

SIN_CLASIFICAR = "sin_clasificar"


class Modelo:
    """Un modelo declarado en la pestania '_modelos', ya resuelto."""

    def __init__(self, fila: dict, metadata: dict):
        self.modelo_id = fila["modelo_id"]
        self.tabla_origen = fila.get("tabla_origen", "")
        self.tabla_destino = fila.get("tabla_destino", "") or self.modelo_id
        self.extractor = fila.get("extractor", "") or "etiqueta_valor"
        self.columna_texto = fila.get("columna_texto", "") or "cuerpo"
        self.filtro = fila.get("filtro", "")
        # El extractor aporta sus campos por defecto y el Sheet los pisa. Asi
        # un modelo con extractor='bac_transacciones' funciona con CERO filas
        # en '_campos', y si el banco cambia una etiqueta se corrige agregando
        # una fila en vez de esperar un despliegue.
        self._extractor = extractores.crear_extractor(self.extractor)
        self.campos_declarados = campos_de(metadata, self.modelo_id)
        self.campos = self._extractor.campos_efectivos(self.campos_declarados)
        self.reglas = clasificacion_de(metadata, self.modelo_id)
        self.overrides = overrides_de(metadata, self.modelo_id)
        self._validar()

    def _validar(self):
        if not self.tabla_origen:
            raise RuntimeError(
                f"El modelo '{self.modelo_id}' no declara 'tabla_origen'."
            )
        if not self.campos:
            raise RuntimeError(
                f"El modelo '{self.modelo_id}' no tiene campos: ni el "
                f"extractor '{self.extractor}' trae un preset, ni hay filas "
                "para este modelo en la pestania '_campos'."
            )
        # Duplicados se validan sobre lo DECLARADO en el Sheet, no sobre el
        # resultado de fusionar con el preset: que una fila del Sheet pise un
        # campo del extractor es el mecanismo esperado (asi se corrige una
        # etiqueta sin desplegar), pero dos filas del Sheet para la misma
        # columna es un error de configuracion que hay que ver.
        vistas = set()
        for campo in self.campos_declarados:
            columna = campo.get("columna", "")
            if columna in vistas:
                raise RuntimeError(
                    f"El modelo '{self.modelo_id}' declara la columna "
                    f"'{columna}' mas de una vez en '_campos'."
                )
            vistas.add(columna)

        for campo in self.campos:
            columna = campo.get("columna", "")
            if not columna:
                raise RuntimeError(
                    f"El modelo '{self.modelo_id}' tiene un campo sin 'columna'."
                )
            if campo.get("tipo") and campo["tipo"] not in tipos.tipos_disponibles():
                raise RuntimeError(
                    f"El modelo '{self.modelo_id}', columna '{columna}': tipo "
                    f"'{campo['tipo']}' desconocido. Disponibles: "
                    f"{', '.join(tipos.tipos_disponibles())}."
                )

    # ---------------- esquema ----------------

    def columnas(self) -> list:
        """
        [(nombre, tipo_logico)] de la tabla derivada.

        Se calcula desde la METADATA y no desde las filas procesadas: un modelo
        cuyo origen todavia no tiene filas que calcen tiene que crear igual su
        tabla, con todas sus columnas. Si no, la tabla aparece el dia que llega
        el primer correo y el bot no la ve hasta entonces.
        """
        salida = [(c, t) for c, t in _TIPOS_TECNICOS.items()]
        vistas = set(_TIPOS_TECNICOS)
        for campo in self.campos:
            for columna, tipo in tipos.columnas_de(campo.get("tipo"),
                                                   campo["columna"]):
                salida.append((columna, tipo))
                vistas.add(columna)
            # La columna de clasificacion se crea aunque nunca se llene: una
            # tabla cuyo esquema depende de si hubo o no clasificacion cambiaria
            # de forma entre corridas, y el catalogo del cliente quedaria
            # describiendo columnas que a veces no existen.
            destino = campo.get("clasifica_en", "")
            if destino and destino not in vistas:
                salida.append((destino, "texto"))
                vistas.add(destino)
        return salida

    def columnas_rechazos(self) -> list:
        return [(COL_CLAVE, "texto"), (COL_ORIGEN, "texto"),
                (COL_MODELO, "texto"), ("motivo", "texto"),
                ("campos_extraidos", "texto")]

    # ---------------- procesamiento ----------------

    def procesar(self, filas_origen: list, mapeo: dict = None) -> tuple:
        """
        Convierte filas raw en (filas_derivadas, rechazos).

        `filas_origen` son dicts de la tabla raw ya filtrados por SQL.
        `mapeo` es {valor_normalizado: valor_asignado} que llena el job del LLM.
        """
        indice_overrides = self._indexar_overrides()
        mapeo = mapeo or {}
        derivadas, rechazos = [], []

        for cruda in filas_origen:
            clave = self._clave_de(cruda)
            texto = cruda.get(self.columna_texto)
            if texto is None:
                rechazos.append(self._rechazo(
                    clave, f"la fila no tiene la columna "
                           f"'{self.columna_texto}' declarada en el modelo", {}))
                continue

            brutos = self._extractor.extraer(str(texto), self.campos)
            fila, faltantes = self._tipar(brutos)

            if faltantes:
                rechazos.append(self._rechazo(
                    clave,
                    "faltan campos requeridos: " + ", ".join(sorted(faltantes)),
                    {k: v for k, v in brutos.items()
                     if not k.startswith("__")},
                ))
                continue

            fila[COL_CLAVE] = clave
            fila[COL_ORIGEN] = self.tabla_origen
            fila[COL_MODELO] = self.modelo_id
            self._clasificar(fila, mapeo, indice_overrides.get(clave, {}))
            derivadas.append(fila)

        return derivadas, rechazos

    def _tipar(self, brutos: dict) -> tuple:
        """Aplica el tipo de cada campo y detecta requeridos faltantes."""
        fila, faltantes = {}, []
        for campo in self.campos:
            columna, tipo = campo["columna"], campo.get("tipo", "texto")
            crudo = brutos.get(columna)
            contexto = {"etiqueta": brutos.get(f"__etiqueta__{columna}", "")}
            convertido = tipos.convertir(tipo, crudo, contexto)

            for sufijo, valor in convertido.items():
                fila[f"{columna}{sufijo}" if sufijo else columna] = valor

            if _es_si(campo.get("requerido")):
                # Requerido mira el valor YA CONVERTIDO, no el texto crudo. Un
                # monto que llego como 'CRC ---' se extrae bien y se convierte
                # a None: la fila tiene que rechazarse igual, porque una
                # transaccion sin monto no sirve para nada.
                if fila.get(columna) in (None, ""):
                    faltantes.append(columna)
        return fila, faltantes

    def _clasificar(self, fila: dict, mapeo: dict, override: dict):
        """
        Precedencia: override > mapeo > regla > sin_clasificar.

        El orden no es negociable y es la razon de ser de las tres capas. Un
        mismo comercio puede corresponder a cuentas distintas segun la compra
        (el super de la semana o un regalo), y esa diferencia NO esta en los
        datos: no hay regla que pueda derivarla. Solo la sabe una persona, y por
        eso el override —que es por transaccion— gana siempre.
        """
        for campo in self.campos:
            destino = campo.get("clasifica_en") or ""
            if not destino:
                continue
            origen = fila.get(campo["columna"])

            if destino in override:
                fila[destino] = override[destino]
                continue

            clave = _normalizar(origen)
            if clave and clave in mapeo:
                fila[destino] = mapeo[clave]
                continue

            fila[destino] = self._por_regla(origen) or SIN_CLASIFICAR

        # Overrides sobre columnas que no son de clasificacion (corregir un
        # monto mal leido, por ejemplo).
        for columna, valor in override.items():
            if columna in fila and fila.get(columna) != valor:
                fila[columna] = valor

    def _por_regla(self, valor):
        """Primera regla que calza, en orden de prioridad."""
        texto = _normalizar(valor)
        if not texto:
            return None
        for regla in self.reglas:
            if _calza_like(texto, _normalizar(regla.get("patron"))):
                return regla.get("valor")
        return None

    def _indexar_overrides(self) -> dict:
        indice = {}
        for o in self.overrides:
            clave = o.get("clave", "").strip()
            columna = o.get("columna", "").strip()
            if clave and columna:
                indice.setdefault(clave, {})[columna] = o.get("valor", "")
        return indice

    def claves_de_override(self) -> set:
        return {o.get("clave", "").strip()
                for o in self.overrides if o.get("clave", "").strip()}

    def _clave_de(self, cruda: dict) -> str:
        """
        Identidad ESTABLE de la fila derivada, que es tambien la llave con la
        que se escribe un override.

        Se prefiere una columna de identidad que ya exista en el origen
        (correo_id, insight_id, evento_id). Eso no es comodidad: esos ids ya son
        estables entre corridas por construccion, o sea que un override escrito
        hoy sigue apuntando a la misma transaccion despues de reconstruir. Si se
        derivara del CONTENIDO extraido, cualquier cambio de formato del banco
        moveria las claves y dejaria todos los overrides huerfanos.
        """
        for candidata in ("correo_id", "insight_id", "evento_id", "id"):
            if cruda.get(candidata):
                return str(cruda[candidata])
        # Sin id natural: hash del contenido de la fila. Funciona, pero es
        # fragil para overrides; por eso se avisa una vez por modelo.
        if not getattr(self, "_aviso_clave", False):
            logger.warning(
                "[%s] la tabla '%s' no tiene una columna de identidad conocida; "
                "la clave se calcula por contenido y los overrides pueden "
                "quedar huerfanos si cambia el formato del origen.",
                self.modelo_id, self.tabla_origen,
            )
            self._aviso_clave = True
        crudo = json.dumps(cruda, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(crudo.encode("utf-8")).hexdigest()

    def _rechazo(self, clave: str, motivo: str, extraidos: dict) -> dict:
        return {
            COL_CLAVE: clave,
            COL_ORIGEN: self.tabla_origen,
            COL_MODELO: self.modelo_id,
            "motivo": motivo,
            "campos_extraidos": json.dumps(
                extraidos, ensure_ascii=False, default=str),
        }

    # ---------------- SQL de lectura ----------------

    def sql_origen(self, esquema: str) -> str:
        """
        SELECT sobre la tabla raw, con el filtro declarado.

        El filtro va tal cual a un WHERE. Es SQL crudo A PROPOSITO: quien
        escribe la metadata es el operador del sistema, no un usuario final, y
        un vocabulario cerrado de condiciones terminaria reinventando SQL peor.
        Esta es la valvula de escape que evita que el resto del vocabulario se
        estire para cubrir casos que no le tocan.
        """
        sql = f'SELECT * FROM "{esquema}"."{self.tabla_origen}"'
        filtro = self.filtro_efectivo()
        if filtro:
            sql += f" WHERE {filtro}"
        return sql

    def filtro_efectivo(self) -> str:
        """El filtro declarado, o el sugerido por el extractor si no hay."""
        return self.filtro.strip() or self._extractor.filtro_sugerido

    def valores_a_clasificar(self) -> list:
        """Columnas cuyo valor alimenta el job de clasificacion del LLM."""
        return [c["columna"] for c in self.campos if c.get("clasifica_en")]


def _normalizar(valor) -> str:
    """
    Normaliza para comparar: sin acentos, mayusculas, espacios colapsados.

    'AUTOMERCADO  Escazú' y 'automercado escazu' tienen que ser el mismo
    comercio, o el mapeo aprendido no sirve de nada y el LLM termina
    clasificando la misma cadena varias veces.
    """
    import unicodedata
    if valor is None:
        return ""
    plano = unicodedata.normalize("NFKD", str(valor))
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    return " ".join(plano.upper().split())


def _calza_like(texto: str, patron: str) -> bool:
    """LIKE de SQL evaluado en Python: % es cualquier cosa, _ es un caracter."""
    import re as _re
    if not patron:
        return False
    regex = "".join(
        ".*" if c == "%" else "." if c == "_" else _re.escape(c)
        for c in patron
    )
    return _re.fullmatch(regex, texto) is not None


def _es_si(valor) -> bool:
    return str(valor).strip().lower() in {"1", "si", "sí", "true", "yes", "x"}
