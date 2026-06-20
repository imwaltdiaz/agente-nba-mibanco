"""
test_abt_schema.py — Tests de integridad de artefactos del Agente Orquestador NBA
===================================================================================
Valida los artefactos producidos por la Iteración 1.1 (Fases A y B):
  - ABT_Train.parquet: granularidad, columnas, tipos, valores válidos.
  - ABT_Inferencia.parquet: granularidad, simetría de features, mora en rango.
  - cate_scores.parquet: esquema exacto del contrato, 0 NaN, cobertura de clientes.

Ejecución:
    pytest tests/test_abt_schema.py -v --tb=short
    pytest tests/test_abt_schema.py -v --tb=short -k "not cate"  # Solo ABTs
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Agregar raíz del proyecto al path para importar src.config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.config as config

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def abt_train() -> pd.DataFrame:
    """Carga ABT_Train.parquet una sola vez por sesión de tests."""
    path = config.INTERIM_DIR / config.ABT_TRAIN_FILE
    if not path.exists():
        pytest.skip(f"ABT_Train.parquet no existe: {path}. Ejecuta data_prep.py primero.")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def abt_inferencia() -> pd.DataFrame:
    """Carga ABT_Inferencia.parquet una sola vez por sesión de tests."""
    path = config.INTERIM_DIR / config.ABT_INFERENCIA_FILE
    if not path.exists():
        pytest.skip(f"ABT_Inferencia.parquet no existe: {path}. Ejecuta data_prep.py primero.")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def cate_scores() -> pd.DataFrame | None:
    """Carga cate_scores.parquet si existe, retorna None si no."""
    path = config.PROCESSED_DIR / config.CATE_SCORES_FILE
    if not path.exists():
        return None
    return pd.read_parquet(path)


def cate_exists() -> bool:
    """Verifica si cate_scores.parquet existe para condicionar tests."""
    return (config.PROCESSED_DIR / config.CATE_SCORES_FILE).exists()


# ---------------------------------------------------------------------------
# TESTS — ABT_TRAIN
# ---------------------------------------------------------------------------

class TestABTTrain:
    """Valida la integridad de ABT_Train.parquet."""

    def test_abt_train_exists(self):
        """ABT_Train.parquet debe existir en data/02_interim/."""
        path = config.INTERIM_DIR / config.ABT_TRAIN_FILE
        assert path.exists(), (
            f"ABT_Train.parquet no encontrado en {path}.\n"
            "  → Ejecuta: python src/data_prep.py"
        )

    def test_abt_train_has_feature_cols(self, abt_train):
        """Todas las FEATURE_COLS definidas en config deben estar en ABT_Train."""
        missing = [f for f in config.FEATURE_COLS if f not in abt_train.columns]
        assert not missing, (
            f"Features ausentes en ABT_Train: {missing}\n"
            f"Columnas disponibles: {list(abt_train.columns)}"
        )

    def test_abt_train_has_target_and_treatment(self, abt_train):
        """ABT_Train debe tener las columnas de target y tratamiento."""
        for col in [config.TARGET_COL, config.TREATMENT_COL, "Deuda_Expuesta", "contacto_id"]:
            assert col in abt_train.columns, f"Columna crítica ausente: '{col}'"

    def test_abt_train_no_nulls_in_critical(self, abt_train):
        """No debe haber NaN en target ni en tratamiento."""
        null_target = abt_train[config.TARGET_COL].isnull().sum()
        null_treatment = abt_train[config.TREATMENT_COL].isnull().sum()
        assert null_target == 0, f"{null_target} NaN en columna '{config.TARGET_COL}'"
        assert null_treatment == 0, f"{null_treatment} NaN en columna '{config.TREATMENT_COL}'"

    def test_abt_train_treatment_values(self, abt_train):
        """canal_contacto debe contener solo valores en {0, 1, 2, 3, 4}."""
        valid_values = set(config.TREATMENT_MAP.values())
        actual_values = set(abt_train[config.TREATMENT_COL].unique())
        invalid = actual_values - valid_values
        assert not invalid, (
            f"Valores inválidos en '{config.TREATMENT_COL}': {invalid}\n"
            f"Valores permitidos: {valid_values}"
        )

    def test_abt_train_has_control_group(self, abt_train):
        """El grupo control sintético (T=0) debe tener al menos 100 filas."""
        n_control = (abt_train[config.TREATMENT_COL] == 0).sum()
        assert n_control >= 100, (
            f"Grupo control insuficiente: {n_control} filas (mínimo 100).\n"
            "  → Verifica build_control_rows() en data_prep.py."
        )

    def test_abt_train_target_is_binary(self, abt_train):
        """pago_7d_post_contacto debe ser estrictamente binario (0 o 1)."""
        valid_values = {0, 1}
        actual_values = set(abt_train[config.TARGET_COL].unique())
        invalid = actual_values - valid_values
        assert not invalid, (
            f"Valores no binarios en '{config.TARGET_COL}': {invalid}\n"
            f"Solo se permiten: {valid_values}"
        )

    def test_abt_train_minimum_size(self, abt_train):
        """ABT_Train debe tener al menos 1000 filas para entrenar un modelo."""
        assert len(abt_train) >= 1000, (
            f"ABT_Train demasiado pequeña: {len(abt_train)} filas."
        )


# ---------------------------------------------------------------------------
# TESTS — ABT_INFERENCIA
# ---------------------------------------------------------------------------

class TestABTInferencia:
    """Valida la integridad de ABT_Inferencia.parquet."""

    def test_abt_inferencia_exists(self):
        """ABT_Inferencia.parquet debe existir en data/02_interim/."""
        path = config.INTERIM_DIR / config.ABT_INFERENCIA_FILE
        assert path.exists(), (
            f"ABT_Inferencia.parquet no encontrado en {path}.\n"
            "  → Ejecuta: python src/data_prep.py"
        )

    def test_abt_inferencia_has_feature_cols(self, abt_inferencia):
        """Todas las FEATURE_COLS deben estar en ABT_Inferencia (simetría con train)."""
        missing = [f for f in config.FEATURE_COLS if f not in abt_inferencia.columns]
        assert not missing, (
            f"Simetría rota: features ausentes en ABT_Inferencia: {missing}\n"
            "  → Verifica build_abt_inferencia() en data_prep.py."
        )

    def test_abt_inferencia_no_target_no_treatment(self, abt_inferencia):
        """ABT_Inferencia NO debe contener el target ni la columna de tratamiento."""
        assert config.TARGET_COL not in abt_inferencia.columns, (
            f"Columna '{config.TARGET_COL}' encontrada en ABT_Inferencia (data leakage)."
        )
        assert config.TREATMENT_COL not in abt_inferencia.columns, (
            f"Columna '{config.TREATMENT_COL}' encontrada en ABT_Inferencia."
        )

    def test_abt_inferencia_unique_clients(self, abt_inferencia):
        """Cada cliente debe aparecer exactamente una vez (granularidad cliente_id)."""
        n_total = len(abt_inferencia)
        n_unique = abt_inferencia["cliente_id"].nunique()
        assert n_total == n_unique, (
            f"Clientes duplicados en ABT_Inferencia: {n_total} filas, {n_unique} únicos.\n"
            "  → La granularidad debe ser 1 fila por cliente_id."
        )

    def test_abt_inferencia_mora_range(self, abt_inferencia):
        """dias_mora debe estar en el rango [DIAS_MORA_MIN, DIAS_MORA_MAX] = [1, 30]."""
        fuera_rango = abt_inferencia[
            (abt_inferencia["dias_mora"] < config.DIAS_MORA_MIN) |
            (abt_inferencia["dias_mora"] > config.DIAS_MORA_MAX)
        ]
        assert len(fuera_rango) == 0, (
            f"{len(fuera_rango)} clientes con dias_mora fuera del rango "
            f"[{config.DIAS_MORA_MIN}, {config.DIAS_MORA_MAX}]:\n"
            f"{fuera_rango[['cliente_id', 'dias_mora']].head(5)}"
        )

    def test_abt_inferencia_has_deuda_expuesta(self, abt_inferencia):
        """Deuda_Expuesta debe existir y ser positiva para todos los clientes."""
        assert "Deuda_Expuesta" in abt_inferencia.columns, (
            "Columna 'Deuda_Expuesta' ausente en ABT_Inferencia."
        )
        n_no_positiva = (abt_inferencia["Deuda_Expuesta"] <= 0).sum()
        assert n_no_positiva == 0, (
            f"{n_no_positiva} clientes con Deuda_Expuesta <= 0."
        )

    def test_abt_inferencia_minimum_size(self, abt_inferencia):
        """ABT_Inferencia debe tener al menos 1 cliente."""
        assert len(abt_inferencia) >= 1, "ABT_Inferencia está vacía."


# ---------------------------------------------------------------------------
# TESTS — CATE SCORES (condicionales: solo si el archivo existe)
# ---------------------------------------------------------------------------

class TestCATEScores:
    """Valida la integridad de cate_scores.parquet (Fase B.2)."""

    @pytest.mark.skipif(not cate_exists(), reason="cate_scores.parquet no existe aún")
    def test_cate_scores_schema(self, cate_scores):
        """cate_scores.parquet debe tener exactamente las 6 columnas del contrato."""
        expected_cols = [
            "Cliente_ID", "Deuda_Expuesta",
            "Uplift_WhatsApp", "Uplift_SMS", "Uplift_Llamada", "Uplift_Campo",
        ]
        assert cate_scores is not None, "cate_scores fixture retornó None inesperadamente."
        missing = [c for c in expected_cols if c not in cate_scores.columns]
        extra = [c for c in cate_scores.columns if c not in expected_cols]
        assert not missing, f"Columnas faltantes en cate_scores: {missing}"
        assert not extra, f"Columnas extra en cate_scores: {extra}"

    @pytest.mark.skipif(not cate_exists(), reason="cate_scores.parquet no existe aún")
    def test_cate_scores_no_nulls(self, cate_scores):
        """cate_scores no debe tener ningún valor NaN."""
        assert cate_scores is not None
        null_counts = cate_scores.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"NaN encontrados en cate_scores:\n{cols_with_nulls}"
        )

    @pytest.mark.skipif(not cate_exists(), reason="cate_scores.parquet no existe aún")
    def test_cate_scores_client_match(self, cate_scores, abt_inferencia):
        """Todos los Cliente_ID en cate_scores deben estar en ABT_Inferencia."""
        assert cate_scores is not None
        inferencia_ids = set(abt_inferencia["cliente_id"].unique())
        scores_ids = set(cate_scores["Cliente_ID"].unique())
        ids_no_match = scores_ids - inferencia_ids
        assert not ids_no_match, (
            f"{len(ids_no_match)} Cliente_ID en cate_scores no presentes en ABT_Inferencia:\n"
            f"Muestra: {list(ids_no_match)[:5]}"
        )

    @pytest.mark.skipif(not cate_exists(), reason="cate_scores.parquet no existe aún")
    def test_cate_scores_deuda_positive(self, cate_scores):
        """Deuda_Expuesta en cate_scores debe ser positiva."""
        assert cate_scores is not None
        n_no_positiva = (cate_scores["Deuda_Expuesta"] <= 0).sum()
        assert n_no_positiva == 0, (
            f"{n_no_positiva} registros con Deuda_Expuesta <= 0 en cate_scores."
        )

    @pytest.mark.skipif(not cate_exists(), reason="cate_scores.parquet no existe aún")
    def test_cate_scores_unique_clients(self, cate_scores):
        """Cada cliente debe aparecer exactamente una vez en cate_scores."""
        assert cate_scores is not None
        n_total = len(cate_scores)
        n_unique = cate_scores["Cliente_ID"].nunique()
        assert n_total == n_unique, (
            f"Clientes duplicados en cate_scores: {n_total} filas, {n_unique} únicos."
        )
