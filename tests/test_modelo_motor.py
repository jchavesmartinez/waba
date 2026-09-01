"""
Pruebas del motor semantico.

Lo que se cuida aca es lo que NO se ve cuando una corrida sale bien: que una
fila que no se pudo extraer quede REGISTRADA en vez de desaparecer, que el
orden de precedencia de la clasificacion se respete, y que la separacion de
monto/moneda y etiqueta/valor sobreviva a los formatos reales del banco.

Los textos de ejemplo son los cuerpos reales de notificaciones del BAC, ya
convertidos a texto plano por el conector IMAP (que es como llegan a la tabla).
"""

import pytest

from modelo.motor import SIN_CLASIFICAR, Modelo, _normalizar
from modelo.tipos import convertir


CUERPO_BAC = """Hola JOSE CARLOS CHAVES MARTINEZ
Comercio: AM PM VEROLIZ
Ciudad y país: HEREDIA, Costa Rica
Fecha: Ago 9, 2026 , 16:22
MASTER: ************8774
Autorización: 328882
Referencia: 90709690
Tipo de Transacción: COMPRA
Monto: CRC 3,320.00"""

# Reenviado a mano: AMEX en vez de MASTER, Referencia VACIA y preambulo de
# Gmail arriba. Los tres detalles vienen de un correo real.
CUERPO_REENVIADO = """---------- Forwarded message ---------
From: Notificación <notificacion@baccredomatic.cr>
Subject: Notificación de transacción NIKE INC E-COMMERCE

Comercio: NIKE INC E-COMMERCE
Ciudad y país: , Estados Unidos
Fecha: Ago 9, 2026 , 17:09
AMEX: ***********8715
Autorización: 873181
Referencia:
Tipo de Transacción: COMPRA
Monto: USD 113.13"""

CUERPO_APLANADO = """BAC Hola JOSE CARLOS CHAVES MARTINEZ A continuacion le detallamos la transaccion realizada: Comercio: HOOTERS Ciudad y pais: HEREDIA, Costa Rica Fecha: Ago 12, 2026 , 19:10 MASTER: ************8774 Autorizacion: 032331 Referencia: 51720096 Tipo de Transaccion: COMPRA Monto: CRC 16,895.07 Tiene dudas sobre esta transaccion?"""

CAMPOS = [
    {"modelo_id": "bac", "columna": "comercio", "tipo": "texto",
     "patron": "Comercio", "requerido": "si",
     "clasifica_en": "cuenta_contable"},
    {"modelo_id": "bac", "columna": "fecha_transaccion",
     "tipo": "fecha_hora_es", "patron": "Fecha", "requerido": "si"},
    {"modelo_id": "bac", "columna": "tarjeta", "tipo": "etiqueta_de_lista",
     "patron": "MASTER,VISA,AMEX"},
    {"modelo_id": "bac", "columna": "referencia", "tipo": "texto",
     "patron": "Referencia"},
    {"modelo_id": "bac", "columna": "monto", "tipo": "monto_con_moneda",
     "patron": "Monto", "requerido": "si"},
]

REGLAS = [
    {"modelo_id": "bac", "patron": "AM PM%", "valor": "Supermercado",
     "prioridad": "10"},
    {"modelo_id": "bac", "patron": "%", "valor": "Gastos varios",
     "prioridad": "99"},
]


def _modelo(overrides=None, reglas=None, campos=None):
    meta = {
        "campos": campos if campos is not None else CAMPOS,
        "clasificacion": reglas if reglas is not None else REGLAS,
        "overrides": overrides or [],
    }
    return Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones", "columna_texto": "cuerpo"},
        meta,
    )


def _fila(cuerpo=CUERPO_BAC, clave="c1"):
    return {"correo_id": clave, "cuerpo": cuerpo,
            "asunto": "Notificacion de compra BAC"}


def test_join_auxiliar_asigna_titular_y_separa_mapeos_por_dimension():
    campos = CAMPOS + [{
        "modelo_id": "bac", "columna": "comercio_concepto",
        "tipo": "texto", "patron": "Comercio",
        "clasifica_en": "linea_presupuesto_id",
        "clasifica_con": "titular",
    }]
    meta = {
        "campos": campos, "clasificacion": [], "overrides": [],
        "joins": [{
            "modelo_id": "bac", "tabla_auxiliar": "sharepoint_db__tarjetas",
            "columna_base": "tarjeta", "columna_auxiliar": "ultimos4",
            "transformacion": "ultimos4", "columnas_salida": "titular",
            "fecha_base": "fecha_transaccion",
            "vigencia_desde": "vigencia_desde",
            "vigencia_hasta": "vigencia_hasta", "activo": "si",
        }],
    }
    modelo = Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones", "columna_texto": "cuerpo"}, meta)
    mapeo = {
        ("cuenta_contable", _normalizar("AM PM VEROLIZ")): "Alimentacion",
        ("linea_presupuesto_id",
         _normalizar("comercio_concepto: AM PM VEROLIZ | titular: Jose")):
            "GAS-008",
    }
    auxiliares = {"sharepoint_db__tarjetas": [{
        "ultimos4": "8774", "titular": "Jose", "activo": "si",
        "vigencia_desde": "2026-01-01", "vigencia_hasta": "",
    }]}
    filas, rechazos = modelo.procesar([_fila()], mapeo, auxiliares)
    assert rechazos == []
    assert filas[0]["titular"] == "Jose"
    assert filas[0]["cuenta_contable"] == "Alimentacion"
    assert filas[0]["linea_presupuesto_id"] == "GAS-008"


def test_join_posterior_usa_linea_presupuesto_nacida_de_clasificacion():
    campos = CAMPOS + [{
        "modelo_id": "bac", "columna": "comercio_concepto",
        "tipo": "texto", "patron": "Comercio",
        "clasifica_en": "linea_presupuesto_id",
        "clasifica_con": "titular",
    }]
    meta = {
        "campos": campos, "clasificacion": [], "overrides": [],
        "joins": [
            {
                "modelo_id": "bac",
                "tabla_auxiliar": "sharepoint_db__tarjetas",
                "columna_base": "tarjeta", "columna_auxiliar": "ultimos4",
                "transformacion": "ultimos4", "columnas_salida": "titular",
                "fecha_base": "fecha_transaccion",
                "vigencia_desde": "vigencia_desde",
                "vigencia_hasta": "vigencia_hasta", "activo": "si",
            },
            {
                "modelo_id": "bac",
                "tabla_auxiliar": "sharepoint_db__presupuesto",
                "columna_base": "linea_presupuesto_id",
                "columna_auxiliar": "linea_id",
                "transformacion": "exacto", "columnas_salida": "concepto",
                "fecha_base": "fecha_transaccion",
                "vigencia_desde": "vigencia_desde",
                "vigencia_hasta": "vigencia_hasta", "activo": "si",
            },
        ],
    }
    modelo = Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones", "columna_texto": "cuerpo"}, meta)
    mapeo = {
        ("cuenta_contable", _normalizar("AM PM VEROLIZ")): "Otros",
        ("linea_presupuesto_id",
         _normalizar("comercio_concepto: AM PM VEROLIZ | titular: Jose")):
            "gas_comedera",
    }
    auxiliares = {
        "sharepoint_db__tarjetas": [{
            "ultimos4": "8774", "titular": "Jose", "activo": "si",
            "vigencia_desde": "2026-01-01", "vigencia_hasta": "",
        }],
        "sharepoint_db__presupuesto": [{
            "linea_id": "gas_comedera", "concepto": "Comedera",
            "categoria": "Alimentacion",
            "activo": "si", "vigencia_desde": "2026-01-01",
            "vigencia_hasta": "",
        }],
    }

    filas, rechazos = modelo.procesar([_fila()], mapeo, auxiliares)

    assert rechazos == []
    assert filas[0]["titular"] == "Jose"
    assert filas[0]["linea_presupuesto_id"] == "gas_comedera"
    assert filas[0]["concepto"] == "Comedera"
    assert filas[0]["cuenta_contable"] == "Alimentacion"


def test_override_de_cuenta_gana_sobre_categoria_de_linea_presupuestaria():
    campos = CAMPOS + [{
        "modelo_id": "bac", "columna": "comercio_concepto",
        "tipo": "texto", "patron": "Comercio",
        "clasifica_en": "linea_presupuesto_id",
    }]
    meta = {
        "campos": campos,
        "clasificacion": [],
        "overrides": [{
            "modelo_id": "bac", "clave": "c1",
            "columna": "cuenta_contable", "valor": "Regalos",
        }],
        "joins": [{
            "modelo_id": "bac",
            "tabla_auxiliar": "sharepoint_db__presupuesto",
            "columna_base": "linea_presupuesto_id",
            "columna_auxiliar": "linea_id",
            "transformacion": "exacto",
            "columnas_salida": "concepto",
            "activo": "si",
        }],
    }
    modelo = Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones", "columna_texto": "cuerpo"},
        meta,
    )
    mapeo = {
        ("linea_presupuesto_id", _normalizar("AM PM VEROLIZ")):
            "gas_comedera",
    }
    auxiliares = {
        "sharepoint_db__presupuesto": [{
            "linea_id": "gas_comedera", "concepto": "Comedera",
            "categoria": "Alimentacion", "activo": "si",
        }],
    }

    filas, rechazos = modelo.procesar([_fila()], mapeo, auxiliares)

    assert rechazos == []
    assert filas[0]["linea_presupuesto_id"] == "gas_comedera"
    assert filas[0]["cuenta_contable"] == "Regalos"


# --- extraccion y tipado --------------------------------------------------

def test_extrae_y_tipa_una_notificacion_real():
    filas, rechazos = _modelo().procesar([_fila()])
    assert rechazos == []
    f = filas[0]
    assert f["comercio"] == "AM PM VEROLIZ"
    assert f["monto"] == 3320.0 and f["monto_moneda"] == "CRC"
    assert f["fecha_transaccion"].hour == 16
    assert f["_clave"] == "c1"


def test_el_separador_de_miles_no_divide_el_gasto_por_mil():
    """'3,320.00' son 3320, no 3.32. El error no rompe nada: solo produce un
    reporte mil veces menor que se ve perfectamente plausible."""
    assert convertir("monto_con_moneda", "CRC 3,320.00")[""] == 3320.0
    assert convertir("monto_con_moneda", "₡3.320,00")[""] == 3320.0
    assert convertir("monto_con_moneda", "USD 113.13")[""] == 113.13


def test_la_fecha_tolera_el_formato_inconsistente_del_banco():
    """Los correos reales traen 'Ago 9, 2026' y 'Ago 9,2026' (sin espacio)."""
    a = convertir("fecha_hora_es", "Ago 9, 2026 , 16:22")[""]
    b = convertir("fecha_hora_es", "Ago 9,2026 , 00:49")[""]
    assert (a.month, a.day, a.hour) == (8, 9, 16)
    assert (b.month, b.day, b.hour) == (8, 9, 0)


def test_la_etiqueta_de_la_tarjeta_es_un_dato_no_un_nombre_fijo():
    """El BAC escribe MASTER:, VISA: o AMEX: segun con que se pago, asi que la
    etiqueta MISMA es informacion y un mapeo etiqueta->columna no alcanza."""
    master, _ = _modelo().procesar([_fila()])
    amex, _ = _modelo().procesar([_fila(CUERPO_REENVIADO, "c2")])
    assert master[0]["tarjeta_tipo"] == "MASTER"
    assert amex[0]["tarjeta_tipo"] == "AMEX"
    assert amex[0]["tarjeta"] == "***********8715"


def test_un_reenvio_manual_se_procesa_igual_que_el_automatico():
    """El cliente puede reenviar a mano y ese correo no deberia perderse."""
    filas, rechazos = _modelo().procesar([_fila(CUERPO_REENVIADO, "c2")])
    assert rechazos == []
    assert filas[0]["comercio"] == "NIKE INC E-COMMERCE"


def test_un_correo_html_aplanado_conserva_todas_las_etiquetas_bac():
    """Zoho puede quitar todos los saltos de linea al convertir cierto HTML."""
    modelo = Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones", "extractor": "bac_transacciones",
         "columna_texto": "cuerpo"},
        {"campos": [], "clasificacion": [], "overrides": []})
    filas, rechazos = modelo.procesar([_fila(CUERPO_APLANADO, "c3")])
    assert rechazos == []
    assert filas[0]["comercio"] == "HOOTERS"
    assert filas[0]["ciudad_pais"] == "HEREDIA, Costa Rica"
    assert filas[0]["tarjeta"] == "************8774"
    assert filas[0]["autorizacion"] == "032331"
    assert filas[0]["referencia"] == "51720096"
    assert filas[0]["tipo_transaccion"] == "COMPRA"
    assert filas[0]["monto"] == 16895.07


def test_un_campo_vacio_no_es_un_campo_ausente():
    """'Referencia:' sin valor es un dato real del BAC en compras
    internacionales; tratarlo como faltante rechazaria una transaccion buena."""
    filas, _ = _modelo().procesar([_fila(CUERPO_REENVIADO, "c2")])
    assert filas[0]["referencia"] == ""


def test_la_tabla_declara_sus_columnas_aunque_no_haya_filas():
    """Si el esquema saliera de las filas procesadas, un modelo sin datos
    todavia crearia una tabla sin columnas y el bot no la veria nunca."""
    columnas = dict(_modelo().columnas())
    for esperada in ("comercio", "cuenta_contable", "monto", "monto_moneda",
                     "tarjeta", "tarjeta_tipo", "_clave"):
        assert esperada in columnas
    assert columnas["monto"] == "decimal"


# --- rechazos: fallar ruidoso ---------------------------------------------

def test_un_correo_que_no_produce_campos_va_a_rechazos_no_al_vacio():
    """
    Es el modo de falla real: cuerpo truncado a 2000 caracteres, cambio de
    formato del banco o cadena de reenvios larga. Los tres se ven identicos a
    'no habia nada', y son problemas que hay que ver.
    """
    filas, rechazos = _modelo().procesar(
        [_fila("Hola, esto no es una notificacion bancaria.", "c9")])
    assert filas == []
    assert len(rechazos) == 1
    assert "comercio" in rechazos[0]["motivo"]
    assert rechazos[0]["_clave"] == "c9"


def test_un_monto_ilegible_rechaza_la_fila():
    """El requerido mira el valor CONVERTIDO: un monto que se extrae pero no se
    puede leer deja una transaccion sin monto, que no sirve para nada."""
    cuerpo = CUERPO_BAC.replace("CRC 3,320.00", "CRC ---")
    _, rechazos = _modelo().procesar([_fila(cuerpo, "c8")])
    assert len(rechazos) == 1 and "monto" in rechazos[0]["motivo"]


# --- clasificacion en tres capas ------------------------------------------

def test_precedencia_override_gana_sobre_mapeo_y_regla():
    """
    El caso que motiva todo el diseño: el MISMO comercio puede ser cuentas
    distintas segun la compra (el super de la semana o un regalo). Esa
    diferencia NO esta en los datos, asi que ninguna regla puede derivarla:
    solo la sabe una persona.
    """
    overrides = [{"modelo_id": "bac", "clave": "c1",
                  "columna": "cuenta_contable", "valor": "Regalos"}]
    filas, _ = _modelo(overrides).procesar(
        [_fila()], mapeo={"AM PM VEROLIZ": "Conveniencia"})
    assert filas[0]["cuenta_contable"] == "Regalos"


def test_la_regla_gana_sobre_un_mapeo_desactualizado():
    filas, _ = _modelo().procesar(
        [_fila()], mapeo={"AM PM VEROLIZ": "Conveniencia"})
    assert filas[0]["cuenta_contable"] == "Supermercado"


def test_el_mapeo_se_usa_si_no_hay_regla():
    filas, _ = _modelo(reglas=[]).procesar(
        [_fila()], mapeo={"AM PM VEROLIZ": "Conveniencia"})
    assert filas[0]["cuenta_contable"] == "Conveniencia"


def test_la_regla_se_aplica_por_prioridad_no_por_orden_del_sheet():
    """'%' calza con todo; sin orden estable podria ganarle a 'AM PM%' segun
    como Google devolviera las filas."""
    filas, _ = _modelo().procesar([_fila()])
    assert filas[0]["cuenta_contable"] == "Supermercado"


def test_sin_ninguna_capa_queda_sin_clasificar_no_adivinado():
    filas, _ = _modelo(reglas=[]).procesar([_fila()])
    assert filas[0]["cuenta_contable"] == SIN_CLASIFICAR


def test_el_mapeo_ignora_acentos_y_espacios_de_mas():
    """Si no, el LLM termina clasificando la misma cadena varias veces."""
    cuerpo = CUERPO_BAC.replace("AM PM VEROLIZ", "Automercado  Escazú")
    filas, _ = _modelo(reglas=[]).procesar(
        [_fila(cuerpo, "c3")], mapeo={"AUTOMERCADO ESCAZU": "Supermercado"})
    assert filas[0]["cuenta_contable"] == "Supermercado"


def test_varias_columnas_forman_un_solo_contexto_de_clasificacion():
    campos = [dict(c) for c in CAMPOS]
    campos[0]["clasifica_con"] = "asunto, tipo_transaccion"
    modelo = _modelo(campos=campos, reglas=[])

    columnas = dict(modelo.columnas())
    assert columnas["asunto"] == "texto"

    cruda = _fila()
    esperada = modelo.valor_clasificacion(
        {"comercio": "AM PM VEROLIZ", "asunto": cruda["asunto"],
         "tipo_transaccion": None}, campos[0])
    filas, _ = modelo.procesar(
        [cruda], mapeo={_normalizar(esperada): "Supermercado"})

    assert filas[0]["asunto"] == "Notificacion de compra BAC"
    assert filas[0]["cuenta_contable"] == "Supermercado"


def test_una_regla_se_puede_contrastar_con_un_mapeo_con_contexto():
    campos = [dict(c) for c in CAMPOS] + [{
        "modelo_id": "bac", "columna": "comercio_concepto",
        "tipo": "texto", "patron": "Comercio",
        "clasifica_en": "linea_presupuesto_id",
        "clasifica_con": "titular,monto",
    }]
    reglas = [{
        "modelo_id": "bac", "patron": "%CHATGPT%", "valor": "gas_chatgpt",
        "prioridad": "10", "clasifica_en": "linea_presupuesto_id",
    }]
    modelo = _modelo(campos=campos, reglas=reglas)
    campo = campos[-1]
    original = modelo.valor_clasificacion(
        {"comercio_concepto": "CHATGPT", "titular": "Aline", "monto": 20},
        campo)

    assert modelo.regla_para_valor_clasificacion(original, campo) == "gas_chatgpt"


def test_una_regla_puede_elegir_las_columnas_que_evalua():
    campos = [dict(c) for c in CAMPOS]
    campos[0]["clasifica_con"] = "asunto,tipo_transaccion"
    reglas = [{"modelo_id": "bac", "columnas": "asunto,tipo_transaccion",
               "patron": "%COMPRA%", "valor": "Compras", "prioridad": "10"}]
    filas, _ = _modelo(campos=campos, reglas=reglas).procesar([_fila()])
    assert filas[0]["cuenta_contable"] == "Compras"


# --- validacion de la metadata --------------------------------------------

def test_dos_campos_con_la_misma_columna_es_error_de_config():
    """Uno pisaria al otro y el ganador dependeria del orden de las filas."""
    campos = CAMPOS + [{"modelo_id": "bac", "columna": "comercio",
                        "tipo": "texto", "patron": "Otro"}]
    with pytest.raises(RuntimeError, match="mas de una vez"):
        _modelo(campos=campos)


def test_un_tipo_inexistente_falla_al_construir_no_en_produccion():
    campos = [{"modelo_id": "bac", "columna": "x", "tipo": "moneda_rara",
               "patron": "X"}]
    with pytest.raises(RuntimeError, match="desconocido"):
        _modelo(campos=campos)


def test_un_modelo_sin_campos_no_arranca():
    with pytest.raises(RuntimeError, match="_campos"):
        _modelo(campos=[])


def test_dos_campos_no_pueden_pisar_la_misma_clasificacion():
    campos = CAMPOS + [
        {"modelo_id": "bac", "columna": "ciudad", "tipo": "texto",
         "patron": "Ciudad", "clasifica_en": "cuenta_contable"}
    ]
    with pytest.raises(RuntimeError, match="clasifica_con"):
        _modelo(campos=campos)


# --- identidad y overrides ------------------------------------------------

def test_la_clave_sale_del_id_del_origen_no_del_contenido():
    """
    Un override apunta a una clave. Si la clave se derivara del contenido
    extraido, cualquier cambio de formato del banco la moveria y dejaria TODAS
    las correcciones huerfanas en silencio.
    """
    filas, _ = _modelo().procesar([_fila()])
    assert filas[0]["_clave"] == "c1"

    cuerpo = CUERPO_BAC.replace("CRC 3,320.00", "CRC 4,000.00")
    otras, _ = _modelo().procesar([_fila(cuerpo, "c1")])
    assert otras[0]["_clave"] == "c1"      # cambio el monto, no la identidad


def test_reconstruir_dos_veces_da_exactamente_lo_mismo():
    """La propiedad que sostiene el modulo: derivada = f(entradas)."""
    modelo = _modelo()
    a, _ = modelo.procesar([_fila()], mapeo={"MASCOTAS": "Mascotas"})
    b, _ = modelo.procesar([_fila()], mapeo={"MASCOTAS": "Mascotas"})
    assert a == b


# --- precision numerica (regresiones encontradas probando de verdad) -------

def test_un_identificador_largo_no_pierde_digitos():
    """
    float64 solo representa enteros exactos hasta 2^53 (~9x10^15). Un
    consecutivo de Hacienda de 20 digitos cae despues de ese limite, y
    convertirlo pasando por float le cambia los ULTIMOS digitos en silencio:
    '00100001010000000123' se guardaba como 100001010000000128. El dato queda
    mal, se ve plausible y no hay ninguna alerta.
    """
    assert convertir("entero", "00100001010000000123")[""] == 100001010000000123
    assert convertir("entero", "9007199254740993")[""] == 9007199254740993


def test_varios_separadores_de_miles_no_anulan_el_numero():
    """'1,234,567' devolvia None y el campo quedaba vacio sin motivo visible."""
    assert convertir("decimal", "1,234,567")[""] == 1234567.0
    assert convertir("monto_con_moneda", "CRC 1,234,567.89")[""] == 1234567.89
    assert convertir("monto_con_moneda", "₡1.234.567,89")[""] == 1234567.89


def test_el_separador_ambiguo_se_lee_como_miles():
    """
    '3,320' podria ser 3 con 320 milesimas. Se toma como miles, que es como se
    escribe aca: leerlo como decimal convertiria un gasto de 3.320 colones en
    3,32 y el reporte quedaria mil veces menor, plausible y sin alerta.
    """
    assert convertir("decimal", "3,320")[""] == 3320.0
    assert convertir("decimal", "120.000")[""] == 120000.0
    # con 1 o 2 decimales no hay ambiguedad
    assert convertir("decimal", "3,32")[""] == 3.32
    assert convertir("decimal", "113.13")[""] == 113.13
