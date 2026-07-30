"""
Configuracion de pytest.

Pone la raiz del repo en sys.path para que las pruebas puedan importar sync,
config, warehouse y sources sin instalar el proyecto como paquete.
"""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

# Valores de mentira para que importar config no exija credenciales reales.
os.environ.setdefault("WAREHOUSE_TIPO", "duckdb")
os.environ.setdefault("WAREHOUSE_DSN", ":memory:")
