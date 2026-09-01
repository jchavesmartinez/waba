"""
Motor de transformacion: raw -> tabla derivada consumible.

    raw_<cliente>.<tabla_origen>
        |  filtro        que filas aplican (SQL declarado en la metadata)
        |  extraccion    texto -> columnas (modelo/extractores.py)
        |  tipado        texto -> valor tipado (modelo/tipos.py)
        |  clasificacion override > regla > mapeo > sin_clasificar
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
import re

from . import extractores, tipos
from .metadata import campos_de, clasificacion_de, joins_de, overrides_de

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
        self.joins = joins_de(metadata, self.modelo_id)
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
        destinos_clasificacion = set()
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
            for contexto in self.columnas_contexto(campo):
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", contexto):
                    raise RuntimeError(
                        f"El modelo '{self.modelo_id}', columna '{columna}': "
                        f"clasifica_con contiene un nombre invalido: '{contexto}'."
                    )
            destino = str(campo.get("clasifica_en", "")).strip()
            if destino:
                if destino in destinos_clasificacion:
                    raise RuntimeError(
                        f"El modelo '{self.modelo_id}' declara mas de un campo "
                        f"que escribe la clasificacion '{destino}'. Usa un solo "
                        "campo principal y agrega los demas en clasifica_con."
                    )
                destinos_clasificacion.add(destino)

        for regla in self.reglas:
            for columna in _lista_columnas(regla.get("columnas")):
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", columna):
                    raise RuntimeError(
                        f"El modelo '{self.modelo_id}' tiene una regla con "
                        f"columna invalida: '{columna}'."
                    )
        for join in self.joins:
            tabla_aux = join.get("tabla_auxiliar", "")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tabla_aux):
                raise RuntimeError(
                    f"El modelo '{self.modelo_id}' tiene tabla auxiliar "
                    f"invalida: '{tabla_aux}'.")
            for nombre in (join.get("columna_base"),
                           join.get("columna_auxiliar")):
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", nombre or ""):
                    raise RuntimeError(
                        f"El modelo '{self.modelo_id}' tiene un _join con "
                        f"columna invalida: '{nombre}'.")
            for nombre in (_lista_columnas(join.get("columnas_salida")) +
                           [join.get("fecha_base"), join.get("vigencia_desde"),
                            join.get("vigencia_hasta")]):
                if nombre and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", nombre):
                    raise RuntimeError(
                        f"El modelo '{self.modelo_id}' tiene un _join con "
                        f"columna invalida: '{nombre}'.")
            if join.get("transformacion", "exacto").lower() not in {
                    "", "exacto", "normalizado", "ultimos4"}:
                raise RuntimeError(
                    f"El modelo '{self.modelo_id}' tiene transformacion de "
                    f"_join desconocida: '{join.get('transformacion')}'.")

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
        # El contexto puede venir de la tabla raw (por ejemplo 'asunto') y no
        # ser un campo extraido del cuerpo. Se copia a la tabla semantica para
        # que el job de clasificacion, que corre despues, pueda leerlo.
        for campo in self.campos:
            for columna in self.columnas_contexto(campo):
                if columna not in vistas:
                    salida.append((columna, "texto"))
                    vistas.add(columna)
        for join in self.joins:
            for columna in _lista_columnas(join.get("columnas_salida")):
                if columna not in vistas:
                    salida.append((columna, "texto"))
                    vistas.add(columna)
        return salida

    def columnas_rechazos(self) -> list:
        return [(COL_CLAVE, "texto"), (COL_ORIGEN, "texto"),
                (COL_MODELO, "texto"), ("motivo", "texto"),
                ("campos_extraidos", "texto")]

    # ---------------- procesamiento ----------------

    def procesar(self, filas_origen: list, mapeo: dict = None,
                 auxiliares: dict = None) -> tuple:
        """
        Convierte filas raw en (filas_derivadas, rechazos).

        `filas_origen` son dicts de la tabla raw ya filtrados por SQL.
        `mapeo` es {valor_normalizado: valor_asignado} que llena el job del LLM.
        """
        indice_overrides = self._indexar_overrides()
        mapeo = mapeo or {}
        auxiliares = auxiliares or {}
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
            self._copiar_contexto_raw(fila, cruda)
            self._aplicar_joins(fila, cruda, auxiliares)
            self._clasificar(fila, mapeo, indice_overrides.get(clave, {}))
            # Algunos joins dependen de una columna que nace de la
            # clasificacion (p. ej. linea_presupuesto_id). La primera pasada
            # mantiene disponibles los joins de contexto como titular para el
            # clasificador; esta segunda completa solamente las salidas que
            # aun faltan, sin repetir ni sobrescribir los joins ya resueltos.
            self._aplicar_joins(
                fila, cruda, auxiliares, solo_si_faltan_salidas=True)
            self._derivar_cuenta_contable_desde_linea(
                fila, cruda, auxiliares, indice_overrides.get(clave, {}))
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
        Precedencia: override > regla > mapeo > sin_clasificar.

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

            # Las reglas son decisiones explicitas y versionadas en la
            # metadata. Deben poder corregir un mapeo aprendido anteriormente
            # por el LLM; de lo contrario, cambiar una regla no reconstruye el
            # historico como promete el modelo semantico.
            por_regla = self._por_regla(fila, campo, origen)
            if por_regla:
                fila[destino] = por_regla
                continue

            valor_clasificacion = self.valor_clasificacion(fila, campo)
            clave = _normalizar(valor_clasificacion)
            clave_mapeo = (destino, clave)
            if clave and clave_mapeo in mapeo:
                fila[destino] = mapeo[clave_mapeo]
                continue
            # Compatibilidad con la tabla _mapeo anterior (una dimension).
            if clave and clave in mapeo:
                fila[destino] = mapeo[clave]
                continue

            fila[destino] = SIN_CLASIFICAR

        # Overrides sobre columnas que no son de clasificacion (corregir un
        # monto mal leido, por ejemplo).
        for columna, valor in override.items():
            if columna in fila and fila.get(columna) != valor:
                fila[columna] = valor

    def _por_regla(self, fila: dict, campo: dict, valor):
        """Primera regla que calza, en orden de prioridad."""
        destino = campo.get("clasifica_en", "")
        principal = next((c.get("clasifica_en") for c in self.campos
                          if c.get("clasifica_en")), "")
        for regla in self.reglas:
            destino_regla = regla.get("clasifica_en", "")
            if destino_regla and destino_regla != destino:
                continue
            if not destino_regla and destino != principal:
                continue
            columnas = _lista_columnas(regla.get("columnas"))
            if columnas:
                texto = _normalizar(" | ".join(
                    str(fila.get(c) or "") for c in columnas
                ))
            else:
                texto = _normalizar(valor)
            if not texto:
                continue
            if _calza_like(texto, _normalizar(regla.get("patron"))):
                return regla.get("valor")
        return None

    def columnas_contexto(self, campo: dict) -> list:
        """Columnas extra que acompanan al valor principal al clasificar."""
        principal = str(campo.get("columna", "")).strip()
        return [c for c in _lista_columnas(campo.get("clasifica_con"))
                if c and c != principal]

    def columnas_de_clasificacion(self, campo: dict) -> list:
        principal = str(campo.get("columna", "")).strip()
        return [principal] + self.columnas_contexto(campo)

    def valor_clasificacion(self, fila: dict, campo: dict) -> str:
        """Texto estable que se mapea y se presenta al LLM."""
        columnas = self.columnas_de_clasificacion(campo)
        if len(columnas) == 1:
            return str(fila.get(columnas[0]) or "").strip()
        return " | ".join(
            f"{columna}: {str(fila.get(columna) or '').strip()}"
            for columna in columnas
        )

    def _copiar_contexto_raw(self, fila: dict, cruda: dict) -> None:
        for campo in self.campos:
            for columna in self.columnas_contexto(campo):
                if columna not in fila:
                    valor = cruda.get(columna)
                    fila[columna] = None if valor is None else str(valor).strip()

    def _aplicar_joins(self, fila: dict, cruda: dict, auxiliares: dict,
                       solo_si_faltan_salidas: bool = False) -> None:
        """Enriquece una fila con tablas auxiliares declaradas en ``_joins``."""
        for join in self.joins:
            salidas = _lista_columnas(join.get("columnas_salida"))
            if (solo_si_faltan_salidas and salidas and
                    all(fila.get(columna) not in (None, "")
                        for columna in salidas)):
                continue
            tabla = join.get("tabla_auxiliar", "")
            base = fila.get(join.get("columna_base"))
            if base is None:
                base = cruda.get(join.get("columna_base"))
            clave = _transformar_join(base, join.get("transformacion"))
            if not clave:
                continue
            candidatas = []
            for aux in auxiliares.get(tabla, []):
                if not _es_si_o_vacio(aux.get("activo", "si")):
                    continue
                otra = _transformar_join(
                    aux.get(join.get("columna_auxiliar")),
                    join.get("transformacion"))
                if otra == clave and _vigente(fila, cruda, aux, join):
                    candidatas.append(aux)
            if len(candidatas) > 1:
                logger.warning(
                    "[%s] _join con %s encontro %d coincidencias para '%s'; "
                    "se usa la primera.", self.modelo_id, tabla,
                    len(candidatas), clave)
            if candidatas:
                for columna in salidas:
                    fila[columna] = candidatas[0].get(columna)

    def _derivar_cuenta_contable_desde_linea(
            self, fila: dict, cruda: dict, auxiliares: dict,
            override: dict) -> None:
        """Deriva la cuenta desde la categoria de la linea presupuestaria.

        ``linea_presupuesto_id`` es mas especifica que ``cuenta_contable``:
        una linea pertenece a una unica categoria del presupuesto. Cuando el
        modelo logra identificarla, esa relacion debe ser la fuente de verdad
        aunque la clasificacion general del comercio haya producido otra
        cuenta. Un override explicito de ``cuenta_contable`` conserva la
        maxima precedencia y no se pisa.
        """
        if "cuenta_contable" in override:
            return

        linea = fila.get("linea_presupuesto_id")
        if linea in (None, "", SIN_CLASIFICAR):
            return

        for join in self.joins:
            if join.get("columna_base") != "linea_presupuesto_id":
                continue
            tabla = join.get("tabla_auxiliar", "")
            clave = _transformar_join(linea, join.get("transformacion"))
            if not clave:
                continue
            for aux in auxiliares.get(tabla, []):
                if not _es_si_o_vacio(aux.get("activo", "si")):
                    continue
                otra = _transformar_join(
                    aux.get(join.get("columna_auxiliar")),
                    join.get("transformacion"))
                if otra != clave or not _vigente(fila, cruda, aux, join):
                    continue
                cuenta = aux.get("cuenta_contable") or aux.get("categoria")
                if cuenta not in (None, ""):
                    fila["cuenta_contable"] = cuenta
                return

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


def _transformar_join(valor, transformacion: str) -> str:
    t = str(transformacion or "exacto").strip().lower()
    if t == "ultimos4":
        return re.sub(r"\D", "", str(valor or ""))[-4:]
    if t == "normalizado":
        return _normalizar(valor)
    return str(valor or "").strip()


def _vigente(fila: dict, cruda: dict, aux: dict, join: dict) -> bool:
    fecha_col = join.get("fecha_base", "")
    if not fecha_col:
        return True
    fecha = fila.get(fecha_col, cruda.get(fecha_col))
    fecha = _a_fecha_comparable(fecha)
    if fecha is None:
        return False
    desde = _a_fecha_comparable(aux.get(join.get("vigencia_desde", "")))
    hasta = _a_fecha_comparable(aux.get(join.get("vigencia_hasta", "")))
    return (desde is None or fecha >= desde) and (hasta is None or fecha <= hasta)


def _a_fecha_comparable(valor):
    if valor is None or valor == "":
        return None
    if hasattr(valor, "date"):
        return valor.date()
    convertido = tipos.convertir("fecha_iso", valor).get("")
    return convertido.date() if convertido else None


def _es_si_o_vacio(valor) -> bool:
    return str(valor or "").strip() == "" or _es_si(valor)


def _lista_columnas(valor) -> list:
    """Lista editable en Sheet: acepta comas o punto y coma."""
    vistas, salida = set(), []
    for parte in re.split(r"[,;]", str(valor or "")):
        columna = parte.strip()
        if columna and columna not in vistas:
            salida.append(columna)
            vistas.add(columna)
    return salida


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
