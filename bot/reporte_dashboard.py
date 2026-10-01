"""PDF descargable del dashboard financiero.

El documento se construye a partir del mismo snapshot autenticado que ve el
usuario. No vuelve a consultar ni expone tablas del warehouse al navegador.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from decimal import Decimal
from xml.sax.saxutils import escape


def _texto(valor: object, limite: int = 180) -> str:
    if valor is None:
        return ""
    if isinstance(valor, (date, datetime)):
        salida = valor.isoformat()
    elif isinstance(valor, Decimal):
        salida = f"{valor:,.2f}"
    elif isinstance(valor, float):
        salida = f"{valor:,.2f}"
    else:
        salida = str(valor)
    salida = " ".join(salida.split())
    return salida if len(salida) <= limite else salida[: limite - 3] + "..."


def _fila_objeto(kpi: dict, fila: list[object]) -> dict[str, object]:
    return {
        str(columna): fila[indice] if indice < len(fila) else ""
        for indice, columna in enumerate(kpi.get("columnas", []))
    }


def crear_pdf(snapshot: dict, saldos: dict) -> bytes:
    """Crea un reporte PDF multipágina, apto para imprimir o compartir."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    pagina = landscape(letter)
    margen = 1.15 * cm
    ancho = pagina[0] - margen * 2
    buffer = io.BytesIO()
    documento = SimpleDocTemplate(
        buffer, pagesize=pagina, leftMargin=margen, rightMargin=margen,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title="Reporte financiero detallado",
        author="Fachavi",
    )
    estilos = getSampleStyleSheet()
    titulo = ParagraphStyle("ReporteTitulo", parent=estilos["Title"], fontSize=19,
                            leading=23, textColor=colors.HexColor("#176B4D"))
    subtitulo = ParagraphStyle("ReporteSubtitulo", parent=estilos["BodyText"],
                               fontSize=9, leading=12, textColor=colors.HexColor("#66716A"))
    seccion = ParagraphStyle("ReporteSeccion", parent=estilos["Heading2"],
                             fontSize=12, leading=16, spaceBefore=12, spaceAfter=6,
                             textColor=colors.HexColor("#176B4D"))
    celda = ParagraphStyle("ReporteCelda", parent=estilos["BodyText"], fontSize=7,
                           leading=8.7, wordWrap="CJK")
    cabecera = ParagraphStyle("ReporteCabecera", parent=celda, fontName="Helvetica-Bold",
                              textColor=colors.white)
    historia = []

    cliente = snapshot.get("cliente", {}) or {}
    periodo = snapshot.get("periodo", {}) or {}
    historia.extend([
        Paragraph("Reporte financiero detallado", titulo),
        Paragraph(
            escape(
                f"{cliente.get('nombre') or 'Cliente'} | "
                f"{periodo.get('etiqueta') or periodo.get('inicio') or ''} | "
                f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
                f"Saldos calculados al: {saldos.get('fecha') or ''}"
            ),
            subtitulo,
        ),
    ])

    def agregar_tabla(nombre: str, columnas: list[str], filas: list[list[object]]) -> None:
        if not filas:
            return
        historia.append(Paragraph(escape(nombre), seccion))
        cantidad = max(len(columnas), 1)
        datos = [[Paragraph(escape(_texto(columna, 70)), cabecera) for columna in columnas]]
        for valores in filas:
            datos.append([
                Paragraph(escape(_texto(valor)), celda)
                for valor in valores[:cantidad]
            ])
        tabla = LongTable(datos, colWidths=[ancho / cantidad] * cantidad,
                          repeatRows=1, hAlign="LEFT", splitByRow=1)
        tabla.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B4D")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D6E0D7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F2")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        historia.append(tabla)

    resumen = next((kpi for kpi in snapshot.get("kpis", [])
                    if kpi.get("kpi") == "presupuesto_disponible"), None)
    if resumen and resumen.get("filas"):
        datos_resumen = _fila_objeto(resumen, resumen["filas"][0])
        agregar_tabla("Resumen mensual", ["Concepto", "Valor"], [
            [str(clave).replace("_", " ").title(), valor]
            for clave, valor in datos_resumen.items()
        ])

    agregar_tabla("Saldos por cuenta", ["Cuenta", "Tipo", "Moneda", "Saldo CRC", "Saldo USD", "Corte inicial"], [
        [cuenta.get("nombre"), cuenta.get("tipo"), cuenta.get("moneda"),
         cuenta.get("saldo_crc"), cuenta.get("saldo_usd") if cuenta.get("tipo") == "credito" else "",
         cuenta.get("fecha_corte")]
        for cuenta in saldos.get("cuentas", [])
    ])

    agregar_tabla("Gastos y movimientos del mes", [
        "Fecha", "Categoría", "Concepto", "Descripción", "Método de pago", "Monto", "Moneda", "Estado",
    ], [
        [str(movimiento.get("fecha") or "")[:10], movimiento.get("categoria") or "Sin clasificar",
         movimiento.get("concepto") or "Gastos sin identificar", movimiento.get("descripcion") or "Movimiento",
         movimiento.get("medio_pago") or "Sin método de pago", movimiento.get("monto"),
         movimiento.get("moneda") or "CRC",
         "Pendiente de sincronización" if movimiento.get("pendiente_sincronizacion") else "Verificado"]
        for movimiento in snapshot.get("movimientos", [])
    ])

    for kpi in snapshot.get("kpis", []):
        if kpi is resumen or not kpi.get("filas"):
            continue
        columnas = [str(columna).replace("_", " ").title() for columna in kpi.get("columnas", [])]
        agregar_tabla(kpi.get("nombre") or str(kpi.get("kpi") or "Detalle"), columnas, kpi["filas"])

    movimientos_cuenta = []
    for cuenta in saldos.get("cuentas", []):
        for movimiento in cuenta.get("movimientos", []):
            movimientos_cuenta.append([
                cuenta.get("nombre"), movimiento.get("fecha"), movimiento.get("descripcion"),
                movimiento.get("monto"), movimiento.get("moneda"), movimiento.get("origen"),
            ])
    agregar_tabla("Movimientos incluidos en los saldos", [
        "Cuenta", "Fecha", "Descripción", "Monto", "Moneda", "Origen",
    ], movimientos_cuenta)

    notas = [["Advertencia", nota] for nota in saldos.get("advertencias", [])]
    notas.extend([
        ["Sin cuenta vinculada", f"{movimiento.get('fecha')} | {movimiento.get('descripcion')} | {movimiento.get('medio_pago')}"]
        for movimiento in saldos.get("sin_vincular", [])
    ])
    agregar_tabla("Notas", ["Tipo", "Detalle"], notas)

    if len(historia) == 2:
        historia.append(Spacer(1, 12))
        historia.append(Paragraph("No hay registros para este período.", subtitulo))

    def pie(canvas, documento_actual):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#66716A"))
        canvas.drawRightString(pagina[0] - margen, 0.55 * cm, f"Página {documento_actual.page}")
        canvas.restoreState()

    documento.build(historia, onFirstPage=pie, onLaterPages=pie)
    return buffer.getvalue()
