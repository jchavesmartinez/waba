"""
Pruebas de la fuente Meta Ads.

Lo que se cuida aca es lo que NO se ve al mirar una corrida exitosa:

  - que releer la ventana corrija los dias en su lugar en vez de duplicarlos,
  - que los dias que salieron de la ventana sobrevivan,
  - que los presupuestos se dividan por la denominacion minima (el error de los
    cien colones convertidos en uno),
  - que renombrar una campania NO parta su historial en dos.
"""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sources.meta_ads import (
    CAMPOS_ESTRUCTURA,
    CAMPOS_INSIGHT,
    MetaAdsSource,
    TIPOS_INSIGHT,
    _campos_culpables,
    _entidad_a_fila,
    _insight_id,
    _texto_desglose,
    hash_insight,
)
from warehouse.base import Corrida
from warehouse.duckdb_dest import DuckDBDestino


CUENTA = {"name": "PYME Demo", "currency": "CRC",
          "timezone_name": "America/Costa_Rica", "currency_offset": 100}


def _corrida(numero: int) -> Corrida:
    return Corrida(
        corrida_id=f"run-{numero}",
        cliente_id="cliente_a",
        fuente_id="pauta_meta",
        tipo="meta_ads",
        inicio=datetime.now(timezone.utc),
    )


def _crudo(fecha="2026-08-03", ad_id="120001", gasto="12500.50",
           compras="3", nombre="Promo agosto"):
    """Respuesta tipica del API para un anuncio en un dia."""
    return {
        "account_id": "1234567890",
        "account_name": "PYME Demo",
        "account_currency": "CRC",
        "campaign_id": "23851",
        "campaign_name": nombre,
        "adset_id": "23852",
        "adset_name": "Heredia 25-45",
        "ad_id": ad_id,
        "ad_name": "Creativo A",
        "objective": "OUTCOME_SALES",
        "date_start": fecha,
        "date_stop": fecha,
        "impressions": "15320",
        "reach": "9877",
        "frequency": "1.5511",
        "spend": gasto,
        "clicks": "412",
        "ctr": "2.6893",
        "cpc": "30.34",
        "cpm": "816.03",
        "inline_link_clicks": "233",
        "actions": [
            {"action_type": "link_click", "value": "233"},
            {"action_type": "purchase", "value": compras},
            {"action_type": "lead", "value": "8"},
            {"action_type":
                "onsite_conversion.messaging_conversation_started_7d",
             "value": "27"},
        ],
        "action_values": [{"action_type": "purchase", "value": "185000"}],
        "video_p25_watched_actions": [
            {"action_type": "video_view", "value": "4120"}],
        "purchase_roas": [{"action_type": "omni_purchase", "value": "14.8"}],
        "quality_ranking": "ABOVE_AVERAGE",
        "attribution_setting": "7d_click,1d_view",
    }


def _fuente(**extra) -> MetaAdsSource:
    conf = {"account_id": "act_1234567890", "token_env": "X"}
    conf.update(extra)
    return MetaAdsSource("pauta_meta", conf)


def _fila(**kwargs) -> dict:
    return _fuente()._a_fila(_crudo(**kwargs), "act_1234567890", "ad", [], CUENTA)


def _df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame([_fila(**kwargs)], columns=list(CAMPOS_INSIGHT))


# --- transformacion -------------------------------------------------------

def test_la_fila_tiene_exactamente_las_columnas_del_contrato():
    assert set(_fila()) == set(CAMPOS_INSIGHT)
    assert set(TIPOS_INSIGHT) == set(CAMPOS_INSIGHT)


def test_las_metricas_dejan_de_ser_texto():
    fila = _fila()
    assert fila["gasto"] == 12500.50
    assert fila["impresiones"] == 15320
    assert fila["clics"] == 412
    # Un numero que llega como string y se guarda como string convierte
    # SUM(gasto) en una concatenacion silenciosa.
    assert isinstance(fila["gasto"], float)
    assert isinstance(fila["impresiones"], int)


def test_las_acciones_se_desdoblan_en_columnas():
    fila = _fila()
    assert fila["compras"] == 3
    assert fila["leads"] == 8
    assert fila["conversaciones_iniciadas"] == 27
    assert fila["clics_enlace"] == 233
    assert fila["valor_compras"] == 185000.0
    # y lo que no tiene columna propia sigue estando entero
    assert "messaging_conversation_started" in fila["acciones"]
    assert "purchase_roas" in fila["raw_insight"]


def test_las_metricas_envueltas_en_lista_se_desenvuelven():
    # video_p25_watched_actions llega como [{'action_type':..,'value':'4120'}]
    assert _fila()["video_25"] == 4120
    assert _fila()["roas_compra"] == 14.8


def test_el_campo_directo_no_se_pierde_cuando_no_hay_accion_equivalente():
    """
    inline_post_engagement llega como campo propio y ademas como action_type.
    Una cuenta sin conversiones configuradas no trae la accion, y el orden de
    aplicacion equivocado dejaba la columna en NULL pese a que el dato SI vino.
    """
    crudo = _crudo()
    crudo["inline_post_engagement"] = "1902"
    crudo["actions"] = [{"action_type": "purchase", "value": "3"}]
    fila = _fuente()._a_fila(crudo, "act_1234567890", "ad", [], CUENTA)
    assert fila["interacciones_publicacion"] == 1902
    assert fila["clics_enlace"] == 233


def test_el_costo_por_resultado_sin_resultados_es_nulo_no_cero():
    fila = _fila(compras="0")
    assert fila["compras"] == 0
    # Un CPA de 0 se leeria como "gratis"; lo correcto es "no aplica".
    assert fila["cpa_compra"] is None
    assert _fila()["cpa_compra"] == pytest.approx(12500.50 / 3, rel=1e-6)


def test_la_zona_horaria_de_la_cuenta_queda_registrada():
    # Meta corta los dias en la zona de la CUENTA. Si no se guarda, una
    # diferencia contra el POS del cliente es imposible de explicar.
    assert _fila()["zona_horaria"] == "America/Costa_Rica"


# --- identidad ------------------------------------------------------------

def test_renombrar_la_campania_no_parte_el_historial():
    antes = _fila(nombre="Promo agosto")
    despues = _fila(nombre="Promo agosto (v2)")
    assert antes["insight_id"] == despues["insight_id"]
    # el nombre si cambia el CONTENIDO, o sea que la fila se corrige
    assert hash_insight(antes) != hash_insight(despues)


def test_dias_y_entidades_distintas_son_filas_distintas():
    ids = {
        _fila()["insight_id"],
        _fila(fecha="2026-08-04")["insight_id"],
        _fila(ad_id="120002")["insight_id"],
    }
    assert len(ids) == 3


def test_el_desglose_es_determinista_y_entra_en_la_identidad():
    crudo = {"age": "25-34", "gender": "female"}
    assert _texto_desglose(crudo, ["gender", "age"]) == "age=25-34;gender=female"
    assert _texto_desglose(crudo, ["age", "gender"]) == "age=25-34;gender=female"

    base = {"cuenta_id": "1", "nivel": "ad", "entidad_id": "9",
            "fecha": date(2026, 8, 3), "fecha_fin": date(2026, 8, 3)}
    uno = _insight_id({**base, "desglose": "age=25-34"})
    otro = _insight_id({**base, "desglose": "age=35-44"})
    assert uno != otro


# --- presupuestos ---------------------------------------------------------

def test_los_presupuestos_se_convierten_desde_la_denominacion_minima():
    fila = _entidad_a_fila(
        "campaign",
        {"id": "23851", "name": "Promo", "status": "ACTIVE",
         "daily_budget": "2500000", "lifetime_budget": "",
         "created_time": "2026-08-01T09:00:00-0600"},
        divisor=100,
        moneda="CRC",
    )
    # ₡25.000, no ₡2.500.000. Sin dividir, el numero sigue siendo plausible y
    # nadie lo nota hasta que el cliente reclama.
    assert fila["presupuesto_diario"] == 25000.0
    assert fila["presupuesto_total"] is None
    assert fila["moneda"] == "CRC"
    assert set(fila) == set(CAMPOS_ESTRUCTURA)


# --- tolerancia a campos depreciados --------------------------------------

def test_se_detecta_el_campo_que_meta_rechaza_para_reintentar_sin_el():
    mensaje = ("Meta rechazo la peticion (codigo 100: (#100) Tried accessing "
               "nonexistent field (social_spend) on node type (AdAccount))")
    assert _campos_culpables(mensaje, ["spend", "social_spend"]) == {"social_spend"}
    # y no puede inventarse un campo que no estabamos pidiendo
    assert _campos_culpables(mensaje, ["spend"]) == set()


# --- resolucion del token -------------------------------------------------

MAPA = '{"cliente_a":"EAAG-token-a","cliente_b":"EAAH-token-b"}'


def test_token_desde_una_variable_dedicada(monkeypatch):
    monkeypatch.setenv("META_ADS_TOKEN_FACHAVI", "EAAG-compartido")
    fuente = _fuente(token_env="META_ADS_TOKEN_FACHAVI")
    assert fuente._token() == "EAAG-compartido"


def test_token_desde_el_mapa_json_compartido(monkeypatch):
    monkeypatch.setenv("META_ADS_TOKENS", MAPA)
    fuente = MetaAdsSource("pauta_meta", {
        "account_id": "act_1", "token_ref": "cliente_b",
    })
    assert fuente._token() == "EAAH-token-b"


def test_el_mapa_se_puede_renombrar(monkeypatch):
    monkeypatch.setenv("TOKENS_META_FACHAVI", MAPA)
    fuente = MetaAdsSource("pauta_meta", {
        "account_id": "act_1", "token_ref": "cliente_a",
        "tokens_env": "TOKENS_META_FACHAVI",
    })
    assert fuente._token() == "EAAG-token-a"


def test_una_clave_mal_escrita_lista_las_disponibles(monkeypatch):
    """Sin esto, un typo en token_ref obliga a ir a mirar la variable a mano."""
    monkeypatch.setenv("META_ADS_TOKENS", MAPA)
    fuente = MetaAdsSource("pauta_meta", {
        "account_id": "act_1", "token_ref": "cliente_c",
    })
    with pytest.raises(RuntimeError) as e:
        fuente._token()
    assert "cliente_a, cliente_b" in str(e.value)


def test_los_errores_del_mapa_nunca_filtran_los_tokens(monkeypatch):
    """
    Un error de la ingesta termina en _meta.sync_corridas y en el log de
    Render. Si el mensaje llevara el contenido de la variable, los tokens de
    todos los clientes quedarian escritos en la bitacora del warehouse.
    """
    monkeypatch.setenv("META_ADS_TOKENS", '{"cliente_a":"EAAG-secreto",,}')
    fuente = MetaAdsSource("pauta_meta", {
        "account_id": "act_1", "token_ref": "cliente_a",
    })
    with pytest.raises(RuntimeError) as e:
        fuente._token()
    assert "EAAG-secreto" not in str(e.value)

    # y tampoco cuando la clave existe pero el mapa no es un objeto
    monkeypatch.setenv("META_ADS_TOKENS", '["EAAG-secreto"]')
    with pytest.raises(RuntimeError) as e:
        fuente._token()
    assert "EAAG-secreto" not in str(e.value)


def test_sin_ninguna_de_las_dos_formas_falla_explicito():
    with pytest.raises(RuntimeError) as e:
        MetaAdsSource("pauta_meta", {"account_id": "act_1"})._token()
    assert "token_env" in str(e.value) and "token_ref" in str(e.value)


def test_declarar_las_dos_gana_la_explicita_y_avisa(monkeypatch, caplog):
    monkeypatch.setenv("META_ADS_TOKEN_FACHAVI", "EAAG-compartido")
    monkeypatch.setenv("META_ADS_TOKENS", MAPA)
    fuente = MetaAdsSource("pauta_meta", {
        "account_id": "act_1",
        "token_env": "META_ADS_TOKEN_FACHAVI",
        "token_ref": "cliente_a",
    })
    with caplog.at_level("WARNING"):
        assert fuente._token() == "EAAG-compartido"
    assert "token_ref" in caplog.text


# --- recorrido completo con el API simulado -------------------------------

class _RespuestaFalsa:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


class _ClienteFalso:
    """Graph API de mentira: dos paginas de insights y una cuenta."""

    llamadas = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None):
        _ClienteFalso.llamadas.append(url)
        if "/insights" in url and "pagina2" not in url:
            return _RespuestaFalsa({
                "data": [_crudo(fecha="2026-08-03")],
                "paging": {"next": "https://graph.facebook.com/pagina2/insights"},
            })
        if "pagina2" in url:
            return _RespuestaFalsa({"data": [_crudo(fecha="2026-08-04")]})
        if "/campaigns" in url:
            return _RespuestaFalsa({"data": [{
                "id": "23851", "name": "Promo agosto", "status": "ACTIVE",
                "effective_status": "ACTIVE", "objective": "OUTCOME_SALES",
                "daily_budget": "2500000", "account_id": "1234567890",
            }]})
        if "/adsets" in url or "/ads" in url:
            return _RespuestaFalsa({"data": []})
        return _RespuestaFalsa(CUENTA)


def test_cargar_produce_las_dos_tablas_con_el_esquema_declarado(monkeypatch):
    import duckdb

    import sources.meta_ads as modulo

    _ClienteFalso.llamadas = []
    monkeypatch.setattr(modulo.httpx, "Client", _ClienteFalso)
    monkeypatch.setenv("META_TOKEN_DEMO", "token-de-mentira")

    fuente = modulo.MetaAdsSource("pauta_meta", {
        "account_id": "1234567890",          # sin act_, se corrige solo
        "token_env": "META_TOKEN_DEMO",
        "incluir_estructura": True,
    })
    con = duckdb.connect(":memory:")
    try:
        frag = fuente.cargar(con)
        # CAST a DATE porque la tabla TEMPORAL de carga queda en TIMESTAMP
        # (pandas), mientras que la del warehouse se crea como DATE segun
        # TIPOS_INSIGHT. Lo que importa es el dia; ver las pruebas de UPSERT,
        # que si leen la tabla persistida y devuelven date.
        insights = con.execute(
            'SELECT CAST(fecha AS DATE), gasto, conversaciones_iniciadas '
            'FROM "meta_ads" ORDER BY fecha'
        ).fetchall()
        columnas = [c[0] for c in con.execute('DESCRIBE "meta_ads"').fetchall()]
        presupuesto = con.execute(
            'SELECT presupuesto_diario FROM "meta_ads_estructura"'
        ).fetchone()
    finally:
        con.close()

    assert frag.tablas == ["meta_ads", "meta_ads_estructura"]
    # la paginacion por cursor se siguio: dos dias, no uno
    assert insights == [
        (date(2026, 8, 3), 12500.50, 27),
        (date(2026, 8, 4), 12500.50, 27),
    ]
    assert columnas == list(CAMPOS_INSIGHT)
    assert presupuesto == (25000.0,)
    # el account_id sin prefijo se normalizo antes de llamar al API
    assert any("act_1234567890/insights" in u for u in _ClienteFalso.llamadas)


# --- persistencia acumulativa ---------------------------------------------

def test_releer_la_ventana_corrige_el_dia_en_vez_de_duplicarlo():
    """La razon de ser del UPSERT: la atribucion mueve las cifras de un dia."""
    destino = DuckDBDestino(":memory:")
    try:
        primero = destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads", _df(compras="3"), _corrida(1),
        )
        # dos dias despues Meta ya atribuyo dos compras mas al MISMO dia
        segundo = destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads", _df(compras="5"), _corrida(2),
        )
        repetido = destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads", _df(compras="5"), _corrida(3),
        )
        filas = destino.conectar().execute(
            'SELECT fecha, compras FROM "raw_cliente_a"."pauta_meta__meta_ads"'
        ).fetchall()
    finally:
        destino.cerrar()

    assert primero["nuevos"] == 1
    assert segundo["actualizados"] == 1
    assert repetido["sin_cambios"] == 1
    # una sola fila para ese dia, con el valor corregido
    assert filas == [(date(2026, 8, 3), 5)]


def test_los_dias_fuera_de_la_ventana_sobreviven():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads",
            _df(fecha="2026-01-15"), _corrida(1),
        )
        # corrida siguiente: la ventana de 30 dias ya no incluye enero
        stats = destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads",
            _df(fecha="2026-08-03"), _corrida(2),
        )
        fechas = destino.conectar().execute(
            'SELECT fecha FROM "raw_cliente_a"."pauta_meta__meta_ads" '
            "ORDER BY fecha"
        ).fetchall()
    finally:
        destino.cerrar()

    assert stats["actual_total"] == 2
    assert fechas == [(date(2026, 1, 15),), (date(2026, 8, 3),)]


def test_la_ventana_vacia_no_borra_el_historial():
    """Una PYME que pausa la pauta no tiene por que perder lo ya guardado."""
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads", _df(), _corrida(1),
        )
        stats = destino.actualizar_insights_meta(
            "raw_cliente_a", "pauta_meta__meta_ads",
            pd.DataFrame([], columns=list(CAMPOS_INSIGHT)), _corrida(2),
        )
    finally:
        destino.cerrar()

    assert stats["actual_total"] == 1
    assert stats["nuevos"] == 0
