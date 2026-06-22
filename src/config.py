"""
config.py — Configuración Central del Agente Orquestador NBA
=============================================================
Centraliza todas las rutas, constantes de negocio, mapeos de tratamiento
y listas de features usadas por los scripts de Fase A y Fase B.
Todos los módulos del proyecto deben importar desde aquí para eliminar
magic strings y garantizar consistencia entre entrenamiento e inferencia.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS DEL PROYECTO
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "01_raw"
INTERIM_DIR = ROOT_DIR / "data" / "02_interim"
PROCESSED_DIR = ROOT_DIR / "data" / "03_processed"
MODELS_DIR = ROOT_DIR / "models"

# ---------------------------------------------------------------------------
# NOMBRES DE ARCHIVOS
# ---------------------------------------------------------------------------
RAW_CLIENTES = "01_Tabla_de_Clientes.csv"
RAW_CREDITOS = "02_Tabla_de_Crditos.csv"
RAW_CONTACTOS = "03_Tabla_contactos.csv"

ABT_TRAIN_FILE = "ABT_Train.parquet"
ABT_INFERENCIA_FILE = "ABT_Inferencia.parquet"
CATE_SCORES_FILE = "cate_scores.parquet"
MODEL_FILE = "uplift_model.joblib"

# ---------------------------------------------------------------------------
# MAPEO DE TRATAMIENTOS
# El grupo Control (T=0) se fabrica sintéticamente en data_prep.py.
# Los 4 canales reales (T=1..4) provienen de la columna canal_contacto
# de la tabla 03_Tabla_contactos.csv.
# ---------------------------------------------------------------------------
TREATMENT_MAP = {
    "control": 0,
    "whatsapp": 1,
    "sms": 2,
    "llamada": 3,
    "campo": 4,
}

# Canales de tratamiento activo (excluye control)
ACTIVE_CHANNELS = ["whatsapp", "sms", "llamada", "campo"]

# Columnas que no entran al modelo causal, pero se arrastran para la
# planificacion operativa posterior: agenda, campo, rutas y reporteria.
OPERATIONAL_COLS = ["region", "zona"]

# ---------------------------------------------------------------------------
# FEATURES DEL MODELO — Lista simétrica usada en TRAIN e INFERENCIA
#
# Incluye:
#   - Features de perfil del cliente (tabla 01)
#   - Features del crédito snapshot (tabla 02)
#   - Features de historial de contacto (tabla 03 en train;
#     imputadas con INFERENCE_DEFAULTS en inferencia)
#   - Features derivadas binarias (creadas en data_prep.py)
#   - Feature temporal de momento del contacto
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # --- Perfil del cliente (tabla 01) ---
    "edad",
    "score_riesgo",
    "prob_default",
    "num_atrasos_previos",
    "dias_mora_promedio",
    "ratio_pago",
    "ultimo_pago_dias",
    "es_digital",
    "uso_app",
    "uso_whatsapp",
    "interaccion_digital_score",
    # Flags de elegibilidad por canal (features predictivas, NO tratamientos)
    "canal_whatsapp",
    "canal_sms",
    "canal_llamada",
    "canal_campo",
    # --- Snapshot del crédito (tabla 02 / tabla 03 en train) ---
    "dias_mora",
    "saldo_restante",
    "cuota_mensual",
    "num_creditos_activos",
    # --- Historial de contacto (tabla 03 en train; imputado en inferencia) ---
    "num_contactos_ult7d",
    "num_contactos_ult30d",
    "dias_ultimo_contacto",
    "intento_num",
    "recency_score",
    "days_since_due",
    # --- Features derivadas binarias (creadas en engineer_client_features) ---
    "genero_M",
    "zona_urbano",
    "tipo_cliente_recurrente",
    # --- Feature temporal (ordinal: 0=Mañana, 1=Tarde, 2=Noche) ---
    "momento_contacto",
]

# ---------------------------------------------------------------------------
# VALORES DE IMPUTACIÓN PARA ABT_INFERENCIA
# Clientes aún no contactados → se asignan valores fácticos, no arbitrarios:
#   - conteos en 0 porque no han sido contactados
#   - dias_ultimo_contacto = 999 (mismo valor usado en datos crudos para "sin_contacto_previo")
#   - days_since_due se llenará con dias_mora del crédito en data_prep
# ---------------------------------------------------------------------------
INFERENCE_DEFAULTS = {
    "num_contactos_ult7d": 0,
    "num_contactos_ult30d": 0,
    "dias_ultimo_contacto": 999,
    "intento_num": 0,
    "recency_score": 0.0,
    # "days_since_due" → se asigna dinámicamente desde dias_mora en data_prep.py
}

# ---------------------------------------------------------------------------
# CONSTANTES DE NEGOCIO
# ---------------------------------------------------------------------------
PERIODO_INFERENCIA = "2026-03"   # Corte de la cartera actual a puntuar
DIAS_MORA_MIN = 1                # Mora temprana: límite inferior (inclusive)
DIAS_MORA_MAX = 30               # Mora temprana: límite superior (inclusive)

TARGET_COL = "pago_7d_post_contacto"   # Variable objetivo binaria (0/1)
TREATMENT_COL = "canal_contacto"        # Variable de tratamiento codificada como int

RANDOM_STATE = 42        # Semilla global para reproducibilidad
TEST_SIZE = 0.2          # Proporción del split test en train_causal.py
MOMENTO_DEFAULT = 1      # Valor neutro de momento_contacto (1 = Tarde)
