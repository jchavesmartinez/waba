"""Menú inicial de WhatsApp para orientar consultas sin fijar tablas en código."""

from bot import catalogo, dashboard, memoria, whatsapp


_MAX_FILAS = 10


def _id_tabla(accion: str, tabla: str) -> str:
    return f"menu:{accion}:tabla:{tabla}"


def _tablas(cliente: dict) -> list:
    ctx = catalogo.construir_contexto(cliente)
    return list(getattr(ctx, "permitidas", []) or [])


def enviar_principal(numero: str, numero_origen: str = "") -> bool:
    return whatsapp.enviar_lista(
        numero,
        "¿Qué deseas hacer?",
        "Elegir opción",
        [{"title": "Opciones", "rows": [
            {"id": "menu:consultar", "title": "Consultar datos",
             "description": "Explora y consulta tus datos"},
            {"id": "menu:editar", "title": "Editar datos",
             "description": "Agregar, modificar o eliminar registros"},
            {"id": "menu:dashboard", "title": "Ver dashboard",
             "description": "Resumen financiero del período"},
            {"id": "menu:libre", "title": "Pregunta libre",
             "description": "Escribe tu pregunta directamente"},
        ]}],
        numero_origen,
    )


def enviar_tablas(cliente: dict, numero: str, accion: str,
                  numero_origen: str = "") -> bool:
    tablas = _tablas(cliente)
    if not tablas:
        return whatsapp.enviar_texto(
            numero, "No encontré tablas habilitadas para esta cuenta.", numero_origen,
        )
    filas = []
    for tabla in tablas[:_MAX_FILAS]:
        nombre = " ".join(tabla.tabla_logica.replace("_", " ").split())
        filas.append({
            "id": _id_tabla(accion, tabla.tabla_logica),
            "title": nombre[:24],
            "description": str(tabla.descripcion or "").strip()[:72] or "Datos disponibles",
        })
    verbo = "consultar" if accion == "consultar" else "editar"
    return whatsapp.enviar_lista(
        numero,
        f"Selecciona la tabla que deseas {verbo}.",
        "Ver tablas",
        [{"title": "Tablas disponibles", "rows": filas}],
        numero_origen,
    )


def manejar_seleccion(cliente: dict, numero: str, seleccion: str,
                      numero_origen: str = "") -> None:
    """Resuelve IDs emitidos por este módulo; ignora los demás de forma segura."""
    if seleccion == "menu:consultar":
        enviar_tablas(cliente, numero, "consultar", numero_origen)
        return
    if seleccion == "menu:editar":
        enviar_tablas(cliente, numero, "editar", numero_origen)
        return
    if seleccion == "menu:dashboard":
        whatsapp.enviar_texto(
            numero, dashboard.mensaje_enlace(cliente, numero, "dashboard"), numero_origen,
        )
        return
    if seleccion == "menu:libre":
        whatsapp.enviar_texto(
            numero, "Perfecto. Escriba su pregunta con sus propias palabras.", numero_origen,
        )
        return

    partes = seleccion.split(":", 3)
    if len(partes) != 4 or partes[0] != "menu" or partes[2] != "tabla":
        whatsapp.enviar_texto(numero, "No reconocí esa opción. Escriba “menú” para empezar.", numero_origen)
        return
    accion, tabla = partes[1], partes[3]
    permitidas = {t.tabla_logica for t in _tablas(cliente)}
    if accion not in {"consultar", "editar"} or tabla not in permitidas:
        whatsapp.enviar_texto(numero, "Esa tabla ya no está disponible. Escriba “menú” para actualizar las opciones.", numero_origen)
        return
    etiqueta = " ".join(tabla.replace("_", " ").split())
    if accion == "consultar":
        memoria.guardar_intercambio(
            cliente, numero,
            f"Seleccionó consultar la tabla {etiqueta} desde el menú.",
            f"Modo consulta seleccionado: la próxima pregunta se refiere a la tabla {etiqueta}.",
            estado={"menu": {"accion": accion, "tabla": tabla}},
        )
        whatsapp.enviar_texto(
            numero,
            f"Consultarás *{etiqueta}*. Escriba qué desea saber de esta tabla.",
            numero_origen,
        )
        return
    # La capa de datos del bot es deliberadamente de solo lectura. No prometemos
    # una escritura que aún no tenga confirmación, autorización ni destino fuente.
    whatsapp.enviar_texto(
        numero,
        f"Seleccionó editar *{etiqueta}*. La edición por WhatsApp se habilitará "
        "cuando esta tabla tenga configurado un origen editable y confirmación de cambios.",
        numero_origen,
    )
