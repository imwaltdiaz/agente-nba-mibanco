# conftest.py — Configuración de pytest para el Agente Orquestador NBA
# Agrega la raíz del proyecto al sys.path para que todos los tests
# puedan importar src.config y otros módulos sin manipulación manual del path.

import sys
from pathlib import Path

# Raíz del proyecto (directorio padre de tests/)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
