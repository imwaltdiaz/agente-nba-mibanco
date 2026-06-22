"""
inference_causal.py - Fase B.2: Inferencia CATE del Agente Orquestador NBA
============================================================================
Responsabilidades:
  1. Carga el modelo TLearner serializado (uplift_model.joblib).
  2. Carga ABT_Inferencia.parquet (cartera actual con mora 1-30 días).
  3. Calcula el CATE (Conditional Average Treatment Effect) individual
     por canal de contacto para cada cliente en la cartera.
  4. Genera y persiste la tabla de scores en data/03_processed/cate_scores.parquet
     con la estructura exacta requerida por el optimizador (Iteración 1.2):
       | Cliente_ID | Deuda_Expuesta | Uplift_WhatsApp | Uplift_SMS | Uplift_Llamada | Uplift_Campo |

Patrón Facade: consume ABT_Inferencia.parquet + uplift_model.joblib.
No realiza feature engineering.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config


# ---------------------------------------------------------------------------
# 1. CARGA DE ABT_INFERENCIA
# ---------------------------------------------------------------------------

def load_abt_inferencia() -> pd.DataFrame:
    """
    Carga la Analytical Base Table de inferencia desde data/02_interim/.

    Retorna:
        pd.DataFrame: ABT_Inferencia con features del cliente (sin target ni tratamiento).

    Raises:
        FileNotFoundError: Si ABT_Inferencia.parquet no existe.
        ValueError: Si faltan columnas críticas (cliente_id o Deuda_Expuesta).
    """
    path = config.INTERIM_DIR / config.ABT_INFERENCIA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}.\n"
            "  -> Ejecuta primero: python src/data_prep.py"
        )

    df = pd.read_parquet(path)
    print(f"  [OK] ABT_Inferencia cargada: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    missing = [c for c in ["cliente_id", "Deuda_Expuesta"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas críticas ausentes en ABT_Inferencia: {missing}\n"
            "  -> Verifica build_abt_inferencia() en data_prep.py."
        )

    return df


# ---------------------------------------------------------------------------
# 2. CARGA DEL MODELO
# ---------------------------------------------------------------------------

def load_model() -> object:
    """
    Carga el modelo TLearner serializado desde models/.

    Retorna:
        object: Modelo EconML TLearner entrenado.

    Raises:
        FileNotFoundError: Si uplift_model.joblib no existe.
    """
    path = config.MODELS_DIR / config.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}.\n"
            "  -> Ejecuta primero: python ./train_causal.py"
        )

    model = joblib.load(path)
    print(f"  [OK] Modelo cargado desde: {path}")
    return model


# ---------------------------------------------------------------------------
# 3. CÁLCULO DE CATE SCORES
# ---------------------------------------------------------------------------

def compute_cate_scores(model: object, df_inf: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el CATE (Conditional Average Treatment Effect) por individuo
    para cada uno de los 4 canales de contacto vs el grupo control.

    La función model.effect() del TLearner de EconML retorna un array
    de shape (n_clientes, n_tratamientos_activos) donde cada columna
    representa el uplift de un canal sobre el control (T=0).

    Args:
        model: TLearner de EconML entrenado con train_causal.py.
        df_inf: ABT_Inferencia con features en las mismas columnas que el train.

    Retorna:
        pd.DataFrame: Tabla de scores con la estructura exacta del contrato:
            | Cliente_ID | Deuda_Expuesta | Uplift_WhatsApp | Uplift_SMS | Uplift_Llamada | Uplift_Campo |
    """
    channel_names = ["WhatsApp", "SMS", "Llamada", "Campo"]

    # Verificar features disponibles
    missing_feats = [f for f in config.FEATURE_COLS if f not in df_inf.columns]
    if missing_feats:
        print(f"  [WARN] Features ausentes en ABT_Inferencia (se imputará con 0): {missing_feats}")

    available_feats = [f for f in config.FEATURE_COLS if f in df_inf.columns]
    X = df_inf[available_feats].copy()

    # Imputar NaN restantes con 0 (no debería haber, pero por seguridad)
    null_counts = X.isnull().sum().sum()
    if null_counts > 0:
        print(f"  [WARN] Imputando {null_counts} valores NaN en X de inferencia con 0.")
        X = X.fillna(0)

    print(f"\n  Calculando CATE para {len(X):,} clientes x {len(available_feats)} features...")
    # ---
    # =====================================================================
    # CAMBIO CRÍTICO: Calcular explícitamente el efecto para cada canal
    # =====================================================================
    cate_test_list = []
    # 1: WhatsApp, 2: SMS, 3: Llamada, 4: Campo
    for t_val in [1, 2, 3, 4]:
        # Calculamos el efecto incremental del canal 't_val' vs Control (0)
        cate_t = model.effect(X.values, T0=0, T1=t_val)
        cate_test_list.append(cate_t)

    # Unir las 4 listas en una sola matriz de shape (n_clientes, 4)
    cate = np.column_stack(cate_test_list)

    # Construir DataFrame de salida (ahora sí sabemos que cate tiene 4 columnas)
    df_scores = pd.DataFrame({
        "Cliente_ID": df_inf["cliente_id"].values,
        "Deuda_Expuesta": df_inf["Deuda_Expuesta"].values,
        "Uplift_WhatsApp": cate[:, 0],
        "Uplift_SMS": cate[:, 1],
        "Uplift_Llamada": cate[:, 2],
        "Uplift_Campo": cate[:, 3],
    })

    # Atributos operativos para la agenda posterior. No se usan como features
    # del modelo, pero permiten programar campo/llamada por region y zona.
    for col in config.OPERATIONAL_COLS:
        if col in df_inf.columns:
            output_col = col.capitalize()
            df_scores[output_col] = df_inf[col].values
    
    # Estadísticas descriptivas
    uplift_cols = ["Uplift_WhatsApp", "Uplift_SMS", "Uplift_Llamada", "Uplift_Campo"]
    print(f"\n  Estadísticas del CATE por canal:")
    print(f"  {'Canal':<18} {'Media':>10} {'Std':>10} {'Mín':>10} {'Máx':>10}")
    print(f"  {'-'*62}")
    for canal, col in zip(channel_names, uplift_cols):
        serie = df_scores[col]
        print(
            f"  {canal:<18} "
            f"{serie.mean():>+10.4f} "
            f"{serie.std():>10.4f} "
            f"{serie.min():>+10.4f} "
            f"{serie.max():>+10.4f}"
        )

    # Canal de mayor uplift por cliente (argmax)
    cate_matrix = df_scores[uplift_cols].values
    best_channel_idx = np.argmax(cate_matrix, axis=1)
    print(f"\n  Canal con mayor uplift por cliente:")
    for i, canal in enumerate(channel_names):
        pct = (best_channel_idx == i).mean() * 100
        n = (best_channel_idx == i).sum()
        print(f"    {canal:<12}: {n:>7,} clientes ({pct:.1f}%)")

    return df_scores


# ---------------------------------------------------------------------------
# 4. MAIN - Orquestador de Fase B.2
# ---------------------------------------------------------------------------

def main():
    """Orquesta la ejecución completa de la Fase B.2: inferencia CATE y generación de scores."""
    print("\n" + "="*60)
    print(" AGENTE ORQUESTADOR NBA - FASE B.2: INFERENCIA CAUSAL")
    print("="*60)

    try:
        # 1. Cargar ABT de inferencia
        print("\n[1/3] Cargando ABT_Inferencia...")
        df_inf = load_abt_inferencia()

        # 2. Cargar modelo entrenado
        print("\n[2/3] Cargando modelo causal...")
        model = load_model()

        # 3. Calcular CATE scores
        print("\n[3/3] Calculando CATE scores por cliente y canal...")
        df_scores = compute_cate_scores(model, df_inf)

        # 4. Persistir resultado
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        scores_path = config.PROCESSED_DIR / config.CATE_SCORES_FILE
        df_scores.to_parquet(scores_path, engine="pyarrow", index=False)

        print(f"\n{'='*60}")
        print("FASE B.2 COMPLETADA [OK]")
        print(f"  CATE scores -> {scores_path}")
        print(f"  Shape: {df_scores.shape[0]:,} filas x {df_scores.shape[1]} columnas")
        print(f"\n  Primeras 5 filas:")
        print(df_scores.head(5).to_string(index=False))
        print(f"\n  Tabla de CATE scores lista para el Optimizador (Iteración 1.2).")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado en inference_causal.py: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
