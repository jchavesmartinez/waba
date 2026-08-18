"""La fecha del bot sigue el reloj del negocio, no el UTC de Render."""

from datetime import datetime, timezone

from bot.tiempo import fecha_local


def test_medianoche_utc_sigue_siendo_dia_anterior_en_costa_rica():
    instante_render = datetime(2026, 8, 18, 0, 8, tzinfo=timezone.utc)
    assert fecha_local(instante_render).isoformat() == "2026-08-17"
