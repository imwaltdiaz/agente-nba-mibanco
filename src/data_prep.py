"""
data_prep.py - Fase A: Preparación de Datos del Agente Orquestador NBA
=======================================================================
Responsabilidades:
  1. Ingesta de las 3 tablas crudas de Mibanco (Clientes, Créditos, Contactos).
  2. Feature engineering (dummies, momento del contacto).
  3. Fabricación del grupo control sintético (client-crédito-meses sin contacto).
  4. Construcción de ABT_Train (granularidad: contacto_id + filas control).
  5. Construcción de ABT_Inferencia (granularidad: cliente_id, mora 1-30 días).
  6. Persistencia de ambas ABTs en formato Parquet en data/02_interim/.

Patrón Facade: expone build_abt_train() y build_abt_inferencia() que devuelven
DataFrames limpios listos para ser consumidos directamente por los scripts de Fase B.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Asegurar que src/ esté en el path para importar config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS CRUDOS
# ---------------------------------------------------------------------------

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga las tres tablas crudas de Mibanco desde el directorio RAW_DIR.

    Retorna:
        tuple: (df_clientes, df_creditos, df_contactos) como DataFrames de Pandas.

    Raises:
        FileNotFoundError: Si alguno de los archivos CSV no existe.
    """
    print(f"\n{'='*60}")
    print("FASE A - Cargando datos crudos desde:")
    print(f"  {config.RAW_DIR}")
    print(f"{'='*60}")

    path_clientes = config.RAW_DIR / config.RAW_CLIENTES
    path_creditos = config.RAW_DIR / config.RAW_CREDITOS
    path_contactos = config.RAW_DIR / config.RAW_CONTACTOS

    for p in [path_clientes, path_creditos, path_contactos]:
        if not p.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p}")

    df_clientes = pd.read_csv(path_clientes)
    print(f"  [OK] Clientes:   {df_clientes.shape[0]:>10,} filas | {df_clientes.shape[1]} columnas")

    df_creditos = pd.read_csv(path_creditos)
    # Convertir fechas de creditos
    for col in ["fecha_corte", "fecha_pago"]:
        if col in df_creditos.columns:
            df_creditos[col] = pd.to_datetime(df_creditos[col], errors="coerce", dayfirst=False)
    # periodo se mantiene como string 'YYYY-MM'
    df_creditos["periodo"] = df_creditos["periodo"].astype(str).str.strip()
    print(f"  [OK] Creditos:   {df_creditos.shape[0]:>10,} filas | {df_creditos.shape[1]} columnas")

    # Contactos: fecha en formato DD/MM/YYYY -> leer como string y convertir
    df_contactos = pd.read_csv(path_contactos, sep=";")
    df_contactos["fecha_contacto"] = pd.to_datetime(
        df_contactos["fecha_contacto"], errors="coerce", dayfirst=True
    )
    print(f"  [OK] Contactos:  {df_contactos.shape[0]:>10,} filas | {df_contactos.shape[1]} columnas")

    return df_clientes, df_creditos, df_contactos


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def extract_moment(hora_str) -> int:
    """
    Clasifica la hora de contacto en un bloque horario ordinal.

    Args:
        hora_str: Hora en formato "HH:MM:SS" (puede ser NaN).

    Retorna:
        int: 0 = Mañana [6-12), 1 = Tarde [12-18), 2 = Noche [18-24) o [0-6).
             Retorna config.MOMENTO_DEFAULT si el valor es inválido.
    """
    try:
        if pd.isna(hora_str):
            return config.MOMENTO_DEFAULT
        hora = int(str(hora_str).split(":")[0])
        if 6 <= hora < 12:
            return 0   # Mañana
        elif 12 <= hora < 18:
            return 1   # Tarde
        else:
            return 2   # Noche
    except (ValueError, TypeError, IndexError):
        return config.MOMENTO_DEFAULT


def engineer_client_features(df_clientes: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece el DataFrame de clientes con variables derivadas binarias.

    Crea:
        - genero_M (int): 1 si el cliente es de género Masculino.
        - zona_urbano (int): 1 si el cliente pertenece a zona urbana.
        - tipo_cliente_recurrente (int): 1 si es cliente recurrente.

    Args:
        df_clientes: DataFrame original de la tabla de clientes.

    Retorna:
        pd.DataFrame: Copia enriquecida (no modifica el original).
    """
    df = df_clientes.copy()
    df["genero_M"] = (df["genero"].str.upper() == "M").astype(int)
    df["zona_urbano"] = (df["zona"].str.lower() == "urbano").astype(int)
    df["tipo_cliente_recurrente"] = (df["tipo_cliente"].str.lower() == "recurrente").astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. CONSTRUCCIÓN DE FILAS TRATADAS (contactos reales)
# ---------------------------------------------------------------------------

def build_treated_rows(
    df_contactos: pd.DataFrame,
    df_clientes_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye las filas tratadas de ABT_Train a partir de los contactos reales.

    Cada fila representa una interacción de contacto con su canal asignado (T=1..4)
    y el outcome binario (pago_7d_post_contacto).

    Args:
        df_contactos: Tabla de contactos históricos.
        df_clientes_feat: Tabla de clientes con features derivadas ya calculadas.

    Retorna:
        pd.DataFrame: Filas tratadas con todas las FEATURE_COLS + tratamiento + target.
    """
    df = df_contactos.copy()

    # Momento del contacto (ordinal 0/1/2)
    df["momento_contacto"] = df["hora_contacto"].apply(extract_moment)

    # Mapear canal_contacto a entero usando TREATMENT_MAP
    df["canal_contacto"] = (
        df["canal_contacto"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(config.TREATMENT_MAP)
    )

    # Deuda Expuesta = cuota mensual del snapshot del crédito en ese momento
    df["Deuda_Expuesta"] = df["cuota_mensual"]

    # Merge con features del cliente (left para conservar todos los contactos)
    cols_cliente = [
        "cliente_id", "edad", "score_riesgo", "prob_default",
        "num_atrasos_previos", "dias_mora_promedio", "ratio_pago",
        "ultimo_pago_dias", "es_digital", "uso_app", "uso_whatsapp",
        "interaccion_digital_score", "canal_whatsapp", "canal_sms",
        "canal_llamada", "canal_campo",
        "genero_M", "zona_urbano", "tipo_cliente_recurrente",
    ]
    df = df.merge(
        df_clientes_feat[cols_cliente],
        on="cliente_id",
        how="left",
    )

    # Seleccionar columnas finales
    output_cols = (
        ["contacto_id", "cliente_id", "credito_id"]
        + config.FEATURE_COLS
        + [config.TREATMENT_COL, "Deuda_Expuesta", config.TARGET_COL]
    )

    # Conservar solo columnas que existen (seguridad ante variaciones del CSV)
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols]


# ---------------------------------------------------------------------------
# 4. CONSTRUCCIÓN DEL GRUPO CONTROL SINTÉTICO
# ---------------------------------------------------------------------------

def build_control_rows(
    df_creditos: pd.DataFrame,
    df_contactos: pd.DataFrame,
    df_clientes_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fabrica filas de control sintético a partir de combinaciones
    (cliente_id, credito_id, periodo) en la tabla de créditos que NO
    tienen ningún contacto registrado en ese mes.

    El outcome del control es pago_realizado_mes (¿pagó sin ser contactado?),
    que funciona como proxy del comportamiento natural de pago sin intervención.

    Args:
        df_creditos: Tabla de créditos panel mensual.
        df_contactos: Tabla de contactos históricos (para identificar meses contactados).
        df_clientes_feat: Tabla de clientes con features derivadas.

    Retorna:
        pd.DataFrame: Filas de control con las mismas columnas que build_treated_rows.
    """
    # Derivar periodo de contacto en formato 'YYYY-MM'
    contactos_periodo = df_contactos["fecha_contacto"].dt.to_period("M").astype(str)

    # Conjunto de claves ya contactadas: (cliente_id, credito_id, periodo)
    contacted_keys = set(
        zip(
            df_contactos["cliente_id"],
            df_contactos["credito_id"],
            contactos_periodo,
        )
    )

    # Crear clave equivalente en tabla de créditos
    df_cred = df_creditos.copy()
    df_cred["_key"] = list(
        zip(df_cred["cliente_id"], df_cred["credito_id"], df_cred["periodo"])
    )

    # Filtrar filas SIN contacto en ese periodo
    mask_no_contacto = ~df_cred["_key"].isin(contacted_keys)
    df_control = df_cred[mask_no_contacto].copy()
    df_control = df_control.drop(columns=["_key"])

    print(f"  >> Pares (cliente, credito, mes) sin contacto: {len(df_control):,}")

    # Generar IDs negativos para distinguir filas de control
    df_control["contacto_id"] = -(np.arange(1, len(df_control) + 1))

    # Variables de tratamiento y target del control
    df_control[config.TREATMENT_COL] = 0   # T = 0 (control)
    df_control["momento_contacto"] = config.MOMENTO_DEFAULT
    df_control[config.TARGET_COL] = df_control["pago_realizado_mes"].fillna(0).astype(int)
    df_control["Deuda_Expuesta"] = df_control["cuota_mensual"]

    # Imputar features de historial de contacto con defaults semánticos
    df_control["num_contactos_ult7d"] = config.INFERENCE_DEFAULTS["num_contactos_ult7d"]
    df_control["num_contactos_ult30d"] = config.INFERENCE_DEFAULTS["num_contactos_ult30d"]
    df_control["dias_ultimo_contacto"] = config.INFERENCE_DEFAULTS["dias_ultimo_contacto"]
    df_control["intento_num"] = config.INFERENCE_DEFAULTS["intento_num"]
    df_control["recency_score"] = config.INFERENCE_DEFAULTS["recency_score"]
    df_control["days_since_due"] = df_control["dias_mora"]  # Valor fáctico del crédito

    # Merge con features del cliente
    cols_cliente = [
        "cliente_id", "edad", "score_riesgo", "prob_default",
        "num_atrasos_previos", "dias_mora_promedio", "ratio_pago",
        "ultimo_pago_dias", "es_digital", "uso_app", "uso_whatsapp",
        "interaccion_digital_score", "canal_whatsapp", "canal_sms",
        "canal_llamada", "canal_campo",
        "genero_M", "zona_urbano", "tipo_cliente_recurrente",
    ]
    df_control = df_control.merge(
        df_clientes_feat[cols_cliente],
        on="cliente_id",
        how="left",
    )

    # Seleccionar columnas finales - mismas que build_treated_rows
    output_cols = (
        ["contacto_id", "cliente_id", "credito_id"]
        + config.FEATURE_COLS
        + [config.TREATMENT_COL, "Deuda_Expuesta", config.TARGET_COL]
    )
    output_cols = [c for c in output_cols if c in df_control.columns]
    return df_control[output_cols]


# ---------------------------------------------------------------------------
# 5. ABT_TRAIN - Tabla Analítica de Entrenamiento
# ---------------------------------------------------------------------------

def build_abt_train(
    df_clientes: pd.DataFrame,
    df_creditos: pd.DataFrame,
    df_contactos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera la Analytical Base Table de entrenamiento del modelo causal.

    Combina filas tratadas (contactos reales, T=1..4) con filas de control
    sintético (client-crédito-meses sin contacto, T=0).
    Granularidad: contacto_id.

    Args:
        df_clientes: Tabla de clientes.
        df_creditos: Tabla de créditos panel mensual.
        df_contactos: Tabla de contactos históricos.

    Retorna:
        pd.DataFrame: ABT_Train lista para el motor causal.
    """
    print(f"\n{'-'*60}")
    print("Construyendo ABT_Train...")

    df_clientes_feat = engineer_client_features(df_clientes)

    # Filas tratadas (contactos reales)
    print("  [1/2] Procesando contactos tratados...")
    treated = build_treated_rows(df_contactos, df_clientes_feat)
    print(f"        >> Filas tratadas:  {len(treated):>10,}")

    # Filas de control sintetico
    print("  [2/2] Fabricando grupo control sintetico...")
    control = build_control_rows(df_creditos, df_contactos, df_clientes_feat)
    print(f"        >> Filas control:   {len(control):>10,}")

    # Concatenar
    abt = pd.concat([treated, control], ignore_index=True)

    # Dropear filas con NaN en TARGET o TREATMENT (integridad mínima)
    n_antes = len(abt)
    abt = abt.dropna(subset=[config.TARGET_COL, config.TREATMENT_COL])
    n_despues = len(abt)
    if n_antes > n_despues:
        print(f"  [WARN] Filas eliminadas por NaN en target/tratamiento: {n_antes - n_despues:,}")

    # Garantizar tipos correctos
    abt[config.TARGET_COL] = abt[config.TARGET_COL].astype(int)
    abt[config.TREATMENT_COL] = abt[config.TREATMENT_COL].astype(int)

    # Reporte de distribuciones
    print(f"\n  ABT_Train final: {len(abt):,} filas x {abt.shape[1]} columnas")
    print("\n  Distribución de tratamientos (canal_contacto):")
    treatment_labels = {v: k for k, v in config.TREATMENT_MAP.items()}
    vc = abt[config.TREATMENT_COL].value_counts().sort_index()
    for t_val, count in vc.items():
        label = treatment_labels.get(t_val, f"T={t_val}")
        print(f"    T={t_val} ({label:<10}): {count:>8,} ({count/len(abt)*100:.1f}%)")

    print("\n  Tasa de conversión (pago_7d_post_contacto) por grupo:")
    for t_val in sorted(abt[config.TREATMENT_COL].unique()):
        mask = abt[config.TREATMENT_COL] == t_val
        tasa = abt.loc[mask, config.TARGET_COL].mean()
        label = treatment_labels.get(t_val, f"T={t_val}")
        print(f"    T={t_val} ({label:<10}): {tasa:.3f}")

    return abt


# ---------------------------------------------------------------------------
# 6. ABT_INFERENCIA - Tabla Analítica de Scoring (cartera actual)
# ---------------------------------------------------------------------------

def build_abt_inferencia(
    df_clientes: pd.DataFrame,
    df_creditos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera la Analytical Base Table para scoring de la cartera actual.

    Filtra clientes con mora temprana (1-30 días) en el periodo de corte
    definido en config.PERIODO_INFERENCIA. Granularidad: cliente_id.
    No contiene variable objetivo ni variable de tratamiento.

    Args:
        df_clientes: Tabla de clientes.
        df_creditos: Tabla de créditos panel mensual.

    Retorna:
        pd.DataFrame: ABT_Inferencia lista para scoring con el modelo causal.
    """
    print("\n" + "-"*60)
    print(f"Construyendo ABT_Inferencia (periodo: {config.PERIODO_INFERENCIA})...")

    # Filtrar créditos del periodo de corte
    df_corte = df_creditos[
        df_creditos["periodo"] == config.PERIODO_INFERENCIA
    ].copy()
    print(f"  Créditos en periodo {config.PERIODO_INFERENCIA}: {len(df_corte):,}")

    # Filtrar mora temprana (1 a 30 días)
    df_mora = df_corte[
        (df_corte["dias_mora"] >= config.DIAS_MORA_MIN) &
        (df_corte["dias_mora"] <= config.DIAS_MORA_MAX)
    ].copy()
    print(f"  Créditos con mora {config.DIAS_MORA_MIN}-{config.DIAS_MORA_MAX} días: {len(df_mora):,}")

    # Agregar a nivel cliente (un cliente puede tener múltiples créditos en mora)
    df_agg = df_mora.groupby("cliente_id").agg(
        Deuda_Expuesta=("cuota_mensual", "sum"),
        dias_mora=("dias_mora", "max"),
        saldo_restante=("saldo_restante", "sum"),
        cuota_mensual=("cuota_mensual", "sum"),
        num_creditos_activos=("credito_id", "count"),
    ).reset_index()
    print(f"  Clientes únicos en mora temprana: {len(df_agg):,}")

    # Enriquecer con features del cliente
    df_clientes_feat = engineer_client_features(df_clientes)
    cols_cliente = [
        "cliente_id", "edad", "score_riesgo", "prob_default",
        "num_atrasos_previos", "dias_mora_promedio", "ratio_pago",
        "ultimo_pago_dias", "es_digital", "uso_app", "uso_whatsapp",
        "interaccion_digital_score", "canal_whatsapp", "canal_sms",
        "canal_llamada", "canal_campo",
        "genero_M", "zona_urbano", "tipo_cliente_recurrente",
    ]
    df_inf = df_agg.merge(
        df_clientes_feat[cols_cliente],
        on="cliente_id",
        how="left",
    )

    # Imputar features de historial de contacto con defaults semánticos
    df_inf["num_contactos_ult7d"] = config.INFERENCE_DEFAULTS["num_contactos_ult7d"]
    df_inf["num_contactos_ult30d"] = config.INFERENCE_DEFAULTS["num_contactos_ult30d"]
    df_inf["dias_ultimo_contacto"] = config.INFERENCE_DEFAULTS["dias_ultimo_contacto"]
    df_inf["intento_num"] = config.INFERENCE_DEFAULTS["intento_num"]
    df_inf["recency_score"] = config.INFERENCE_DEFAULTS["recency_score"]
    df_inf["days_since_due"] = df_inf["dias_mora"]  # Valor fáctico del crédito

    # Momento del contacto: valor neutro (Tarde = 1)
    df_inf["momento_contacto"] = config.MOMENTO_DEFAULT

    # Seleccionar columnas finales: cliente_id + FEATURE_COLS + Deuda_Expuesta
    output_cols = ["cliente_id"] + config.FEATURE_COLS + ["Deuda_Expuesta"]
    output_cols = [c for c in output_cols if c in df_inf.columns]
    df_inf = df_inf[output_cols]

    # Reporte
    print(f"\n  ABT_Inferencia final: {len(df_inf):,} filas x {df_inf.shape[1]} columnas")
    print("\n  Estadísticas de Deuda_Expuesta (cuota mensual en mora):")
    stats = df_inf["Deuda_Expuesta"].describe()
    print(f"    Media: S/ {stats['mean']:>10.2f} | Mediana: S/ {df_inf['Deuda_Expuesta'].median():>10.2f}")
    print(f"    Mín:   S/ {stats['min']:>10.2f} | Máx:     S/ {stats['max']:>10.2f}")
    print(f"    Deuda total expuesta: S/ {df_inf['Deuda_Expuesta'].sum():,.2f}")

    return df_inf


# ---------------------------------------------------------------------------
# 7. MAIN - Orquestador de Fase A
# ---------------------------------------------------------------------------

def main():
    """Orquesta la ejecución completa de la Fase A: carga, limpieza y construcción de ABTs."""
    print("\n" + "="*60)
    print(" AGENTE ORQUESTADOR NBA - FASE A: PREPARACION DE DATOS")
    print("="*60)

    try:
        # 1. Cargar datos crudos
        df_clientes, df_creditos, df_contactos = load_raw_data()

        # 2. Construir ABT_Train
        abt_train = build_abt_train(df_clientes, df_creditos, df_contactos)

        # 3. Construir ABT_Inferencia
        abt_inferencia = build_abt_inferencia(df_clientes, df_creditos)

        # 4. Persistir como Parquet
        os.makedirs(config.INTERIM_DIR, exist_ok=True)

        train_path = config.INTERIM_DIR / config.ABT_TRAIN_FILE
        inf_path = config.INTERIM_DIR / config.ABT_INFERENCIA_FILE

        abt_train.to_parquet(train_path, engine="pyarrow", index=False)
        abt_inferencia.to_parquet(inf_path, engine="pyarrow", index=False)

        print(f"\n{'='*60}")
        print("FASE A COMPLETADA [OK]")
        print(f"  ABT_Train      -> {train_path}")
        print(f"                    {len(abt_train):,} filas x {abt_train.shape[1]} columnas")
        print(f"  ABT_Inferencia -> {inf_path}")
        print(f"                    {len(abt_inferencia):,} filas x {abt_inferencia.shape[1]} columnas")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
        print("  Verifica que los CSVs estén en data/01_raw/")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado en data_prep.py: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
