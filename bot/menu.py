"""Menú inicial de WhatsApp para orientar consultas sin fijar tablas en código."""

from bot import catalogo, dashboard, edicion, memoria, whatsapp


_MAX_FILAS = 10


def _id_tabla(accion: str, tabla: str) -> str:
    return f"menu:{accion}:tabla:{tabla}"


def _id_accion_edicion(tabla: str, accion: str) -> str:
    return f"menu:editar:accion:{tabla}:{accion}"


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
    if accion == "editar":
        tablas = [t for t in tablas if edicion.politica_desde_tabla(t)]
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

    accion_partes = seleccion.split(":", 4)
    if (len(accion_partes) == 5 and accion_partes[:3] == ["menu", "editar", "accion"]):
        tabla, accion_edicion = accion_partes[3], accion_partes[4]
        politica = edicion.politica_para(cliente, tabla)
        if not politica or accion_edicion not in politica.acciones:
            whatsapp.enviar_texto(numero, "Esa acción ya no está disponible. Escriba “menú” para actualizar las opciones.", numero_origen)
            return
        etiqueta = " ".join(tabla.replace("_", " ").split())
        estado = {"edicion": {"tabla": tabla, "accion": accion_edicion,
                                "paso": "campos", "valores": {}}}
        memoria.guardar_intercambio(
            cliente, numero,
            f"Seleccionó {accion_edicion} en la tabla {etiqueta} desde el menú.",
            f"Edición iniciada: {accion_edicion} en {etiqueta}.", estado=estado,
        )
        if accion_edicion == "anular":
            mensaje = ("Indique el identificador del registro a anular, por ejemplo:\n"
                       f"*{politica.campos[politica.clave_primaria].etiqueta}:* MAN-20260903-AB12CD34")
        elif accion_edicion == "modificar":
            mensaje = ("Indique el identificador y los campos a cambiar, uno por línea. Por ejemplo:\n"
                       f"*{politica.campos[politica.clave_primaria].etiqueta}:* MAN-20260903-AB12CD34\n*Monto:* 12500")
        else:
            requeridos = [c.etiqueta for c in politica.campos.values()
                          if c.requerido and not c.calculado]
            mensaje = ("Envíe los datos como `Campo: valor`, uno por línea. "
                       "Los campos obligatorios son: " + ", ".join(requeridos) + ".")
        whatsapp.enviar_texto(numero, mensaje, numero_origen)
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
    politica = edicion.politica_para(cliente, tabla)
    if not politica:
        whatsapp.enviar_texto(
            numero, "Esa tabla no está habilitada para edición.", numero_origen,
        )
        return
    acciones = list(politica.acciones)
    if len(acciones) <= 3:
        whatsapp.enviar_botones(
            numero,
            f"¿Qué desea hacer con *{etiqueta}*?",
            [{"id": _id_accion_edicion(tabla, accion_edicion),
              "title": accion_edicion.capitalize()}
             for accion_edicion in acciones],
            numero_origen,
        )
        return
    # Meta admite solo tres botones: preservamos el menú de lista si una
    # política futura declarara más acciones.
    filas = [{"id": _id_accion_edicion(tabla, accion_edicion),
              "title": accion_edicion.capitalize(), "description": "Acción permitida"}
             for accion_edicion in acciones[:_MAX_FILAS]]
    whatsapp.enviar_lista(numero, f"¿Qué desea hacer con *{etiqueta}*?", "Elegir acción",
                          [{"title": "Acciones", "rows": filas}], numero_origen)
