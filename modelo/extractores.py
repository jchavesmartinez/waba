"""
Registro de TIPOS de extractor + factory.

Mismo patron que sources/__init__.py. Para agregar un extractor:
  1. Subclase de Extractor con `tipo = "mi_extractor"`.
  2. Registrarla abajo con el decorador @registrar.
  3. Las filas de '_modelos' con extractor="mi_extractor" ya lo usan.

DOS FAMILIAS DE EXTRACTOR, y la diferencia decide si el sistema escala o no:

  GENERICOS (etiqueta_valor, regex, json_ruta)
      Saben leer una FORMA de texto, no un documento concreto. La metadata dice
      que buscar. Un caso nuevo con la misma forma es cero codigo: filas en el
      Sheet y listo.

  CON NOMBRE (bac_transacciones, ...)
      Son PRESETS sobre un generico: heredan el mecanismo y aportan las
      etiquetas y el filtro que ya conocemos de ese documento. Dar de alta un
      cliente del BAC pasa a ser UNA fila en '_modelos', sin escribir '_campos'.

El preset es un atajo, NO un fork. Un extractor con nombre no reimplementa la
extraccion: si lo hiciera, cada banco seria codigo nuevo y estariamos donde
estabamos antes de tener este modulo. Si un documento no se puede leer con
ningun generico, lo que corresponde es agregar un GENERICO nuevo (una forma
mas), no un extractor por cliente.

Los campos del preset se pueden pisar desde el Sheet: si el BAC cambia una
etiqueta, se arregla agregando filas en '_campos' para ese modelo, sin esperar
un despliegue.
"""

import json
import logging
import re
import unicodedata

logger = logging.getLogger("fachavi.modelo.extractores")

_REGISTRO = {}


def registrar(cls):
    """Registra una clase de extractor por su atributo `tipo`."""
    if not getattr(cls, "tipo", None) or cls.tipo == "base":
        raise ValueError(f"El extractor {cls} debe definir un 'tipo' propio.")
    _REGISTRO[cls.tipo] = cls
    return cls


def crear_extractor(tipo: str):
    tipo = (tipo or "etiqueta_valor").strip().lower()
    if tipo not in _REGISTRO:
        raise RuntimeError(
            f"Tipo de extractor desconocido: '{tipo}'. Disponibles: "
            f"{tipos_disponibles()}."
        )
    return _REGISTRO[tipo]()


def tipos_disponibles() -> list:
    return sorted(_REGISTRO)


def _plegar(texto: str) -> str:
    """
    Minusculas, sin acentos, espacios colapsados: para comparar etiquetas.

    'Ciudad y país' y 'Ciudad y pais' tienen que ser la misma etiqueta. Quien
    escribe la metadata no deberia tener que reproducir los acentos exactos del
    correo, y un solo acento distinto haria fallar la extraccion sin dar
    ninguna pista de por que.
    """
    plano = unicodedata.normalize("NFKD", str(texto))
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    return " ".join(plano.lower().split())


def _sin_acentos(texto: str) -> str:
    plano = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in plano if not unicodedata.combining(c))


class Extractor:
    """
    Contrato: texto -> {columna: texto_crudo}.

    Devuelve SIEMPRE texto sin convertir. La conversion a numero, fecha o
    moneda la hace tipos.py. La separacion importa porque asi el mismo
    extractor sirve para campos de cualquier tipo y el mismo tipo sirve para
    cualquier extractor; si se mezclaran, cada combinacion nueva seria codigo
    nuevo.

    No decide si la fila es valida (eso es 'requerido' en el motor), no
    clasifica y no toca la base.
    """

    tipo = "base"

    # Preset: campos que el extractor ya conoce. Los de '_campos' los pisan.
    campos_por_defecto = ()
    # Filtro sugerido para '_modelos'. Informativo: NO se aplica solo, porque
    # un WHERE invisible en el SQL seria imposible de depurar desde el Sheet.
    filtro_sugerido = ""

    def extraer(self, texto: str, campos: list) -> dict:
        raise NotImplementedError

    def campos_efectivos(self, campos_metadata: list) -> list:
        """
        Combina el preset con lo declarado en el Sheet. Gana el Sheet.

        Asi una etiqueta que cambia en el banco se arregla agregando una fila,
        sin esperar a que alguien despliegue codigo.
        """
        por_columna = {c["columna"]: dict(c) for c in self.campos_por_defecto}
        orden = [c["columna"] for c in self.campos_por_defecto]
        for campo in campos_metadata:
            columna = campo.get("columna")
            if not columna:
                continue
            if columna not in por_columna:
                orden.append(columna)
            # Solo pisa lo que la fila del Sheet trae CON VALOR: una celda
            # vacia no deberia borrar el preset sin querer.
            util = {k: v for k, v in campo.items() if v not in ("", None)}
            por_columna[columna] = {**por_columna.get(columna, {}), **util}
        return [por_columna[c] for c in orden]


# --------------------------------------------------------------------------
# Genericos
# --------------------------------------------------------------------------

@registrar
class EtiquetaValor(Extractor):
    """
    Pares "Etiqueta: valor", uno por linea.

        Comercio: NIKE INC E-COMMERCE
        Fecha: Ago 9, 2026 , 17:09
        AMEX: ***********8715

    Cubre notificaciones bancarias, confirmaciones de pedido y facturas en
    texto: cualquier documento con pares etiqueta/valor.

    DECISIONES QUE PARECEN DETALLES Y NO LO SON:

    1. Se indexa TODO el texto una vez y despues se buscan las etiquetas
       pedidas, en vez de recorrer el texto una vez por campo. Ademas de ser
       mas rapido, evita que el ORDEN de los campos en el Sheet cambie el
       resultado.

    2. Gana la PRIMERA aparicion de cada etiqueta. En un correo reenviado con
       cadena de respuestas, lo mas reciente esta arriba; quedarse con la
       ultima devolveria los datos de un correo viejo del hilo.

    3. Un valor VACIO se conserva como cadena vacia y no se descarta.
       'Referencia:' sin valor es un dato real del BAC en compras
       internacionales, y tratarlo como "no encontrado" haria que un campo
       requerido rechazara una transaccion perfectamente valida.

    4. La etiqueta declarada puede ser una LISTA separada por comas. Eso
       resuelve el caso de la tarjeta: el BAC escribe 'MASTER:', 'VISA:' o
       'AMEX:' segun con que se pago, o sea que la etiqueta MISMA es el dato.
       Con "MASTER,VISA,AMEX" se captura la que aparezca y el tipo
       'etiqueta_de_lista' la guarda en su propia columna.
    """

    tipo = "etiqueta_valor"

    def extraer(self, texto: str, campos: list) -> dict:
        indice = self._indexar(texto)
        # Algunos proveedores de correo convierten el HTML a texto sin
        # conservar los <br>: todo el aviso termina en UNA sola linea
        # ("... Comercio: X Ciudad y pais: Y Fecha: ..."). En ese formato el
        # indexador historico no puede separar pares. Se completa el indice
        # buscando exclusivamente las etiquetas declaradas, nunca palabras
        # arbitrarias del cuerpo.
        if len(str(texto or "").splitlines()) <= 1:
            for clave, valor in self._indexar_aplanado(texto, campos).items():
                indice.setdefault(clave, valor)
        salida = {}
        for campo in campos:
            columna = campo.get("columna")
            if not columna:
                continue
            for etiqueta in _etiquetas_de(campo):
                clave = _plegar(etiqueta.rstrip(":"))
                if clave in indice:
                    salida[columna] = indice[clave]
                    salida[f"__etiqueta__{columna}"] = etiqueta.rstrip(":")
                    break
        return salida

    @staticmethod
    def _indexar_aplanado(texto: str, campos: list) -> dict:
        """Extrae pares conocidos de un cuerpo que perdio los saltos de linea."""
        original = str(texto or "")
        if not original:
            return {}

        # _sin_acentos conserva una posicion por caracter para el alfabeto
        # usado por BAC; asi los offsets encontrados sirven para cortar el
        # texto ORIGINAL y no perdemos tildes ni el numero enmascarado.
        plegado = _sin_acentos(original).upper()
        etiquetas = []
        for campo in campos:
            for etiqueta in _etiquetas_de(campo):
                limpia = etiqueta.rstrip(":").strip()
                if limpia:
                    etiquetas.append((_plegar(limpia), limpia))
        por_clave = {clave: original_etq for clave, original_etq in etiquetas}
        if not por_clave:
            return {}

        alternativas = sorted(por_clave, key=len, reverse=True)
        patron = re.compile(
            r"(?<![A-Z0-9_])(" + "|".join(re.escape(e) for e in alternativas)
            + r")\s*:", re.IGNORECASE)
        marcas = list(patron.finditer(plegado))
        indice = {}
        for i, marca in enumerate(marcas):
            clave = _plegar(marca.group(1))
            fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(original)
            valor = original[marca.end():fin].strip()
            if clave not in indice:  # igual que el camino multilínea
                indice[clave] = valor
        return indice

    @staticmethod
    def _indexar(texto: str) -> dict:
        """
        {etiqueta_plegada: valor} recorriendo el texto una sola vez.

        El limite de 60 caracteres para la etiqueta no es arbitrario: sin el,
        cualquier linea con dos puntos se vuelve un par. Una URL o una frase
        larga generarian etiquetas basura que podrian pisar a una real.
        """
        indice = {}
        for linea in str(texto or "").splitlines():
            linea = linea.strip()
            if not linea or ":" not in linea:
                continue
            etiqueta, _, valor = linea.partition(":")
            etiqueta = etiqueta.strip()
            if not etiqueta or len(etiqueta) > 60:
                continue
            clave = _plegar(etiqueta)
            if clave not in indice:                  # gana la primera
                indice[clave] = valor.strip()
        return indice


@registrar
class Regex(Extractor):
    """Un patron por campo, con el valor en el primer grupo de captura."""

    tipo = "regex"

    def extraer(self, texto: str, campos: list) -> dict:
        salida = {}
        texto = str(texto or "")
        for campo in campos:
            columna, patron = campo.get("columna"), campo.get("patron", "")
            if not columna or not patron:
                continue
            try:
                m = re.search(patron, texto, re.IGNORECASE | re.MULTILINE)
            except re.error as e:
                # Un patron mal escrito en el Sheet no puede tumbar la corrida
                # de un cliente entero. Se salta el campo; si ademas era
                # requerido, las filas caen en rechazos con el motivo, o sea
                # que el problema aparece con nombre y apellido.
                logger.error("patron invalido en la columna '%s': %s", columna, e)
                continue
            if m:
                salida[columna] = (m.group(1) if m.groups() else m.group(0)).strip()
        return salida


@registrar
class JsonRuta(Extractor):
    """
    Campos dentro de una columna JSON, con rutas tipo 'cliente.direccion.canton'.

    Util para las columnas crudas que ya guardan varios conectores
    (raw_insight, raw_evento, adjuntos).
    """

    tipo = "json_ruta"

    def extraer(self, texto: str, campos: list) -> dict:
        try:
            datos = json.loads(texto) if isinstance(texto, str) else texto
        except (TypeError, ValueError):
            return {}
        if not isinstance(datos, (dict, list)):
            return {}

        salida = {}
        for campo in campos:
            columna, ruta = campo.get("columna"), campo.get("patron", "")
            if not columna or not ruta:
                continue
            valor = datos
            for parte in str(ruta).split("."):
                if isinstance(valor, dict):
                    valor = valor.get(parte)
                elif isinstance(valor, list) and parte.isdigit():
                    valor = valor[int(parte)] if int(parte) < len(valor) else None
                else:
                    valor = None
                if valor is None:
                    break
            if valor is not None:
                # Un objeto anidado se guarda como JSON, no como repr de
                # Python: "{'a': 1}" no lo puede leer nadie despues.
                salida[columna] = (
                    json.dumps(valor, ensure_ascii=False)
                    if isinstance(valor, (dict, list)) else str(valor)
                )
        return salida


# --------------------------------------------------------------------------
# Con nombre (presets)
# --------------------------------------------------------------------------

@registrar
class BacTransacciones(EtiquetaValor):
    """
    Notificaciones de transaccion de BAC Credomatic (Costa Rica).

    Hereda el mecanismo de EtiquetaValor y aporta lo unico propio del
    documento: sus etiquetas. Dar de alta un cliente del BAC es UNA fila en
    '_modelos' con extractor='bac_transacciones'; no hace falta escribir
    '_campos' salvo para pisar algo.

    Los campos salen de correos reales, y cada uno tiene su motivo:

      - 'tarjeta' declara MASTER,VISA,AMEX porque la etiqueta cambia segun con
        que se pago. Con tipo 'etiqueta_de_lista' queda la marca en
        'tarjeta_tipo' y el numero enmascarado en 'tarjeta'.
      - 'autorizacion' y 'referencia' son TEXTO, no numero: '000382' pierde los
        ceros de la izquierda como entero, y 'AMWD1020' ni siquiera es numerico.
      - 'referencia' NO es requerida: en compras internacionales viene vacia.
      - 'monto' usa monto_con_moneda porque llega como 'CRC 3,320.00'. La misma
        cuenta mezcla colones y dolares, asi que la moneda tiene que quedar en
        su propia columna o sumar el gasto no significa nada.
      - 'tipo_transaccion' distingue COMPRA de CARGO AUTOMATICO, y mas adelante
        de REVERSO o ANULACION. Sin esa columna una compra revertida se cuenta
        dos veces y el libro no cuadra.

    El filtro sugerido mira el CUERPO y no el remitente a proposito: cuando el
    cliente reenvia a mano, el remitente es su propio correo y el del banco
    queda dentro del texto.
    """

    tipo = "bac_transacciones"
    filtro_sugerido = "cuerpo ILIKE '%baccredomatic%'"

    campos_por_defecto = (
        {"columna": "comercio", "tipo": "texto", "patron": "Comercio",
         "requerido": "si", "clasifica_en": "cuenta_contable",
         "descripcion": "Nombre del comercio segun lo reporta el banco."},
        {"columna": "ciudad_pais", "tipo": "texto", "patron": "Ciudad y pais",
         "descripcion": "Ciudad y pais del comercio. La ciudad puede venir "
                        "vacia en compras por internet."},
        {"columna": "fecha_transaccion", "tipo": "fecha_hora_es",
         "patron": "Fecha", "requerido": "si",
         "descripcion": "Momento de la transaccion en hora local de Costa "
                        "Rica. NO es la fecha del correo: el aviso puede "
                        "llegar horas despues."},
        {"columna": "tarjeta", "tipo": "etiqueta_de_lista",
         "patron": "MASTER,VISA,AMEX",
         "descripcion": "Ultimos digitos de la tarjeta usada."},
        {"columna": "autorizacion", "tipo": "texto", "patron": "Autorizacion",
         "descripcion": "Numero de autorizacion del banco."},
        {"columna": "referencia", "tipo": "texto", "patron": "Referencia",
         "descripcion": "Referencia del banco. Puede venir vacia."},
        {"columna": "tipo_transaccion", "tipo": "texto",
         "patron": "Tipo de Transaccion", "requerido": "si",
         "descripcion": "COMPRA, CARGO AUTOMATICO, REVERSO, ANULACION. Los "
                        "reversos y anulaciones DEVUELVEN dinero: no se suman "
                        "junto con las compras."},
        {"columna": "monto", "tipo": "monto_con_moneda", "patron": "Monto",
         "requerido": "si",
         "descripcion": "Monto de la transaccion. La moneda queda en "
                        "'monto_moneda'; la misma cuenta tiene transacciones "
                        "en colones y en dolares."},
    )


def _etiquetas_de(campo: dict) -> list:
    """La etiqueta declarada, que puede ser una lista separada por comas."""
    crudo = campo.get("patron") or campo.get("columna") or ""
    return [p.strip() for p in str(crudo).split(",") if p.strip()]
