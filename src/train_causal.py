"""
train_causal.py - Fase B.1: Entrenamiento del Modelo Causal del Agente Orquestador NBA
========================================================================================
Responsabilidades:
  1. Carga ABT_Train.parquet generada por data_prep.py (patrón Facade: sin ETL aquí).
  2. Split estratificado 80/20 por tratamiento para mantener proporciones.
  3. Entrena un TLearner (EconML) con base LGBMClassifier sobre el set de train.
  4. Evalúa la calidad del CATE sobre el test set: ATE por canal + tabla de deciles.
  5. Serializa el modelo entrenado en models/uplift_model.joblib.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

def load_abt_train() -> pd.DataFrame:
    """
    Carga la Analytical Base Table de entrenamiento desde data/02_interim/.

    Retorna:
        pd.DataFrame: ABT_Train con features, tratamiento y target.

    Raises:
        FileNotFoundError: Si ABT_Train.parquet no existe.
        ValueError: Si faltan columnas críticas de tratamiento o target.
    """
    path = config.INTERIM_DIR / config.ABT_TRAIN_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}.\n"
            "  -> Ejecuta primero: python src/data_prep.py"
        )

    df = pd.read_parquet(path)
    print(f"  [OK] ABT_Train cargada: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    missing = [c for c in [config.TARGET_COL, config.TREATMENT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas críticas ausentes en ABT_Train: {missing}")

    return df


# ---------------------------------------------------------------------------
# 2. PREPARACIÓN DE MATRICES
# ---------------------------------------------------------------------------

def prepare_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extrae las matrices X (features), T (tratamiento) e Y (target) del DataFrame.

    Verifica que el grupo control (T=0) esté presente y que las features
    definidas en config.FEATURE_COLS existan en el DataFrame.

    Args:
        df: ABT_Train cargada.

    Retorna:
        tuple: (X, T, Y) como DataFrame y Series de Pandas.

    Raises:
        ValueError: Si faltan features o no hay grupo control.
    """
    # Verificar features
    missing_feats = [f for f in config.FEATURE_COLS if f not in df.columns]
    if missing_feats:
        print(f"  [WARN] Features ausentes en ABT_Train (se omitirán): {missing_feats}")

    available_feats = [f for f in config.FEATURE_COLS if f in df.columns]

    X = df[available_feats].copy()
    T = df[config.TREATMENT_COL].astype(int)
    Y = df[config.TARGET_COL].astype(int)

    # Validar presencia del grupo control
    if 0 not in T.values:
        raise ValueError(
            "El grupo control (T=0) no está presente en ABT_Train.\n"
            "  -> Verifica la función build_control_rows() en data_prep.py."
        )

    # Rellenar NaN en X con mediana por columna
    null_counts = X.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        print(f"  [WARN] Features con NaN (se imputará con mediana): {list(cols_with_nulls.index)}")
        X = X.fillna(X.median(numeric_only=True))

    print(f"\n  Forma de X: {X.shape}")
    print(f"  Features usadas: {len(available_feats)}")
    print("\n  Distribución del tratamiento (T):")
    treatment_labels = {v: k for k, v in config.TREATMENT_MAP.items()}
    vc = T.value_counts().sort_index()
    for t_val, count in vc.items():
        label = treatment_labels.get(t_val, f"T={t_val}")
        tasa_y = Y[T == t_val].mean()
        print(f"    T={t_val} ({label:<10}): {count:>8,} muestras | tasa_pago={tasa_y:.3f}")

    return X, T, Y


# ---------------------------------------------------------------------------
# 3. SPLIT TRAIN / TEST
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    T: pd.Series,
    Y: pd.Series,
) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento (80%) y evaluación (20%)
    estratificando por la variable de tratamiento T.

    Args:
        X: Matriz de features.
        T: Variable de tratamiento.
        Y: Variable objetivo.

    Retorna:
        tuple: (X_train, X_test, T_train, T_test, Y_train, Y_test)
    """
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=T,
    )

    print(f"\n  Split 80/20 estratificado por tratamiento:")
    print(f"    Train: {len(X_train):>8,} muestras")
    print(f"    Test:  {len(X_test):>8,} muestras")

    print("\n  Distribución de T en train:")
    treatment_labels = {v: k for k, v in config.TREATMENT_MAP.items()}
    for t_val in sorted(T_train.unique()):
        n = (T_train == t_val).sum()
        label = treatment_labels.get(t_val, f"T={t_val}")
        print(f"    T={t_val} ({label:<10}): {n:>8,}")

    return X_train, X_test, T_train, T_test, Y_train, Y_test


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO DEL MODELO CAUSAL
# ---------------------------------------------------------------------------

def calculate_ipw(
    X: pd.DataFrame,
    T: pd.Series,
) -> np.ndarray:
    """
    Calcula los pesos de Inverse Probability Weighting (IPW) para corregir
    el sesgo de selección histórico de los asesores.

    Entrena un LGBMClassifier rápido (propensity score model) para estimar
    la probabilidad de que cada observación haya recibido tratamiento activo
    (T > 0). Las probabilidades se clipa en [0.05, 0.95] para estabilidad
    numérica y los pesos se calculan como w = 1 / P(T=1 | X).

    Args:
        X: Matriz de features de entrenamiento.
        T: Variable de tratamiento (0 = control, >0 = activo).

    Retorna:
        np.ndarray: Array de pesos IPW de longitud len(X).
    """
    from lightgbm import LGBMClassifier

    # Variable binaria: 1 si el cliente recibió algún tratamiento activo
    T_binary = (T > 0).astype(int)

    propensity_model = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    propensity_model.fit(X.values, T_binary.values)

    # P(T_activo=1 | X): probabilidad de recibir tratamiento activo
    propensities = propensity_model.predict_proba(X.values)[:, 1]

    # Clipping para estabilidad numérica (evita pesos explosivos)
    propensities = np.clip(propensities, 0.05, 0.95)

    # w = 1 / P(T=1 | X): mayor peso a observaciones sub-representadas
    weights = 1.0 / propensities

    print(
        f"  [IPW] Propensity scores calculados:"
        f" media={propensities.mean():.3f}"
        f" | min={propensities.min():.3f}"
        f" | max={propensities.max():.3f}"
    )
    print(
        f"  [IPW] Pesos IPW:"
        f" media={weights.mean():.3f}"
        f" | max={weights.max():.3f}"
    )
    return weights


class CustomTLearner:
    """
    Implementación manual de un T-Learner (Causal Inference) que permite
    utilizar pesos por muestra (sample_weight) en cada modelo base.
    """
    def __init__(self, models: dict):
        self.models = models

    def effect(self, X, T0=0, T1=1) -> np.ndarray:
        # Predecir con el modelo del tratamiento T1
        pred_T1 = self.models[T1].predict(X)
        # Predecir con el modelo del tratamiento T0 (control)
        pred_T0 = self.models[T0].predict(X)
        return pred_T1 - pred_T0


def train_uplift_model(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    Y_train: pd.Series,
) -> object:
    """
    Entrena un TLearner manual (CustomTLearner) con base LGBMRegressor aplicando IPW
    para corregir el sesgo de selección histórico de los asesores.
    """
    from lightgbm import LGBMRegressor

    # --- TAREA 1: Calcular pesos IPW para corrección de sesgo ---
    print("  [IPW] Calculando Propensity Scores para corrección de sesgo...")
    sample_weights = calculate_ipw(X_train, T_train)

    models = {}
    canales = np.unique(T_train.values)
    
    print(f"  [CustomTLearner] Entrenando sub-modelos independientes por canal: {canales}")
    for canal in canales:
        mask = (T_train == canal)
        X_canal = X_train[mask].values
        Y_canal = Y_train[mask].values
        w_canal = sample_weights[mask]

        print(f"    -> Canal {canal}: {len(X_canal):,} muestras")

        base_model = LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            verbose=-1,
        )

        base_model.fit(
            X_canal,
            Y_canal,
            sample_weight=w_canal,
        )
        models[canal] = base_model

    print("  [OK] Entrenamiento completado (con corrección IPW manual).")
    return CustomTLearner(models)


# ---------------------------------------------------------------------------
# 5. EVALUACIÓN DEL MODELO
# ---------------------------------------------------------------------------

def calcular_auuc_analitico(cate_predicho, y_real, t_real):
    """
    Calcula el Área Bajo la Curva de Uplift (AUUC / Qini) para validar el modelo.
    Compara el desempeño del modelo para ordenar clientes frente a una asignación al azar.
    """
    df_auuc = pd.DataFrame({'cate': cate_predicho, 'Y': y_real, 'T': t_real})
    # Ordenar de mayor a menor Uplift predicho
    df_auuc = df_auuc.sort_values(by='cate', ascending=False).reset_index(drop=True)
    
    # Conteos y pagos acumulados
    df_auuc['n_tratados'] = (df_auuc['T'] > 0).astype(int).cumsum()
    df_auuc['n_control'] = (df_auuc['T'] == 0).astype(int).cumsum()
    df_auuc['pagos_tratados'] = (df_auuc['Y'] * (df_auuc['T'] > 0)).cumsum()
    df_auuc['pagos_control'] = (df_auuc['Y'] * (df_auuc['T'] == 0)).cumsum()
    
    # Evitar división por cero
    n_t = df_auuc['n_tratados'].replace(0, 1)
    n_c = df_auuc['n_control'].replace(0, 1)
    
    # Fórmula de Uplift Acumulado
    df_auuc['uplift_acum'] = df_auuc['pagos_tratados'] - df_auuc['pagos_control'] * (n_t / n_c)
    
    # Cálculo del área (Qini)
    max_uplift = df_auuc['uplift_acum'].iloc[-1]
    area_modelo = df_auuc['uplift_acum'].sum()
    area_azar = (max_uplift * len(df_auuc)) / 2
    
    coeficiente_qini = area_modelo / area_azar if area_azar != 0 else 0.0
    return min(max(coeficiente_qini, 0.0), 1.0)


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    T_test: pd.Series,
    Y_test: pd.Series,
) -> dict:
    
    channel_names = ["WhatsApp", "SMS", "Llamada", "Campo"]

    # Calcular explícitamente el efecto de cada Tratamiento vs Control
    cate_test_list = []
    for t_val in [1, 2, 3, 4]:
        cate_t = model.effect(X_test.values, T0=0, T1=t_val)
        cate_test_list.append(cate_t)
        
    # Unir las 4 predicciones en una matriz de shape (n_test, 4)
    cate_test = np.column_stack(cate_test_list)

    print(f"\n  {'-'*50}")
    print("  EVALUACIÓN DEL MODELO CAUSAL EN TEST SET")
    print(f"  {'-'*50}")

    metrics = {}

    print("\n  ATE (Average Treatment Effect) por canal vs Control:")
    for i, canal in enumerate(channel_names):
        if i < cate_test.shape[1]:
            ate = cate_test[:, i].mean()
            pct_positivo = (cate_test[:, i] > 0).mean() * 100
        else:
            ate = 0.0
            pct_positivo = 0.0
        metrics[f"ATE_{canal}"] = ate
        metrics[f"pct_positivo_{canal}"] = pct_positivo
        print(f"    {canal:<10}: ATE={ate:>+.4f} | {pct_positivo:.1f}% clientes con CATE > 0")

    # Tabla de deciles para el canal con mayor ATE
    ate_values = [metrics[f"ATE_{c}"] for c in channel_names]
    best_canal_idx = np.argmax(ate_values)
    best_canal = channel_names[best_canal_idx]
    
    if best_canal_idx < cate_test.shape[1]:
        cate_best = cate_test[:, best_canal_idx]
    else:
        cate_best = np.zeros(len(X_test))

    print(f"\n  Tabla de Deciles de Uplift - Canal: {best_canal} (mayor ATE)")
    print(f"  {'Decil':<8} {'CATE_medio':>12} {'Tasa_conv_obs':>15} {'N_muestras':>12}")
    print(f"  {'-'*50}")

    df_eval = pd.DataFrame({
        "cate": cate_best,
        "Y": Y_test.values,
    })
    df_eval["decil"] = pd.qcut(df_eval["cate"], q=10, labels=False, duplicates="drop")

    decil_stats = df_eval.groupby("decil").agg(
        cate_medio=("cate", "mean"),
        tasa_conv=("Y", "mean"),
        n=("Y", "count"),
    ).reset_index()

    for _, row in decil_stats.iterrows():
        print(
            f"  {int(row['decil'])+1:<8} "
            f"{row['cate_medio']:>+12.4f} "
            f"{row['tasa_conv']:>15.3f} "
            f"{int(row['n']):>12,}"
        )

    print(f"\n  ℹ Si la tasa de conversión observada es más alta en deciles")
    print(f"    superiores de CATE predicho, el modelo es predictivo y útil.")

    # --- INCORPORACIÓN DE LA MÉTRICA QINI / AUUC ---
    # Filtramos la data real solo para el grupo Control (0) y el Mejor Canal
    # El Tratamiento 1 corresponde al index 0, Tratamiento 2 al index 1, etc.
    t_objetivo = best_canal_idx + 1
    mask_qini = (T_test == 0) | (T_test == t_objetivo)
    
    cate_qini = cate_best[mask_qini]
    y_qini = Y_test[mask_qini].values
    t_qini = T_test[mask_qini].values
    
    qini_score = calcular_auuc_analitico(cate_qini, y_qini, t_qini)
    metrics[f"Qini_{best_canal}"] = qini_score
    
    print(f"\n  [MÉTRICA CAUSAL] Coeficiente Qini / AUUC ({best_canal} vs Control): {qini_score:.3f}")
    if qini_score > 0.5:
        print("    -> ¡Modelo Altamente Predictivo! El modelo supera ampliamente al azar.")
    elif qini_score > 0.2:
        print("    -> Modelo con buen poder predictivo y separación de clases.")
    else:
        print("    -> Modelo con bajo poder predictivo.")

    return metrics

# ---------------------------------------------------------------------------
# 6. MAIN - Orquestador de Fase B.1
# ---------------------------------------------------------------------------

def main():
    """Orquesta la ejecución completa de la Fase B.1: entrenamiento y evaluación causal."""
    print("\n" + "="*60)
    print(" AGENTE ORQUESTADOR NBA - FASE B.1: ENTRENAMIENTO CAUSAL")
    print("="*60)

    try:
        # 1. Cargar datos
        print("\n[1/5] Cargando ABT_Train...")
        df = load_abt_train()

        # 2. Preparar matrices
        print("\n[2/5] Preparando matrices X, T, Y...")
        X, T, Y = prepare_matrices(df)

        # 3. Split train / test
        print("\n[3/5] Realizando split 80/20 estratificado...")
        X_train, X_test, T_train, T_test, Y_train, Y_test = split_data(X, T, Y)

        # 4. Entrenar modelo
        print("\n[4/5] Entrenando modelo causal...")
        model = train_uplift_model(X_train, T_train, Y_train)

        # 5. Evaluar en test set
        print("\n[5/5] Evaluando modelo en test set...")
        metrics = evaluate_model(model, X_test, T_test, Y_test)

        # 6. Serializar modelo
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model_path = config.MODELS_DIR / config.MODEL_FILE
        joblib.dump(model, model_path)

        print(f"\n{'='*60}")
        print("FASE B.1 COMPLETADA [OK]")
        print(f"  Modelo guardado -> {model_path}")
        print(f"  Tamaño del archivo: {model_path.stat().st_size / 1024:.1f} KB")
        print("\n  Resumen ATEs:")
        for canal in ["WhatsApp", "SMS", "Llamada", "Campo"]:
            print(f"    {canal:<10}: ATE={metrics[f'ATE_{canal}']:>+.4f}")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Validación fallida: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado en train_causal.py: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
