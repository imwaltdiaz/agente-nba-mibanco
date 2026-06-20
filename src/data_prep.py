"""
Fase A: Preparación de Datos
Este script cruza las tablas originales de Mibanco (Clientes, Créditos, Contactos)
y construye las Master Tables (ABT_Train y ABT_Inferencia) para los modelos.
"""

import os
import pandas as pd
import numpy as np

# Configuración de Rutas
RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/01_raw")
INTERIM_DIR = os.path.join(os.path.dirname(__file__), "../data/02_interim")

def load_raw_data():
    """
    Carga las tablas crudas originales en DataFrames.
    """
    print(f"Cargando archivos desde {RAW_DIR}...")
    df_clientes = pd.read_csv(os.path.join(RAW_DIR, "01_Tabla_de_Clientes.csv"))
    df_creditos = pd.read_csv(os.path.join(RAW_DIR, "02_Tabla_de_Crditos.csv"))
    df_contactos = pd.read_csv(os.path.join(RAW_DIR, "03_Tabla_contactos.csv"))
    return df_clientes, df_creditos, df_contactos

def extract_timing_moment(hora_str):
    """
    Clasifica la hora de contacto en los bloques correspondientes:
    - Mañana (08:00 - 12:00)
    - Tarde (12:00 - 17:00)
    - Noche (17:00 - 21:00)
    
    Parámetros:
        hora_str (str): Hora en formato HH:MM:SS
    """
    # TODO: Implementar lógica de agrupación horaria por los otros DS
    # Ejemplo sugerido:
    # hora = int(hora_str.split(':')[0])
    # if 8 <= hora < 12: return 'Mañana'
    # elif 12 <= hora < 17: return 'Tarde'
    # else: return 'Noche'
    return 'Mañana'

def preprocess_features(df_clientes, df_creditos):
    """
    Limpia y transforma las variables del cliente y del crédito.
    """
    # TODO: Limpieza de nulos, imputaciones, codificación de variables categóricas
    # (género, región, zona, tipo_cliente) y normalizaciones.
    pass

def build_abt_train(df_clientes, df_creditos, df_contactos):
    """
    Genera la Analytical Base Table (ABT) para entrenamiento del modelo causal.
    Copia a nivel de interacción (contacto realizado) con el outcome: pago_7d_post_contacto.
    
    Retorna:
        pd.DataFrame: ABT_Train lista para el modelo causal.
    """
    print("Construyendo ABT_Train...")
    # TODO: Cruzar contactos con clientes y estado de créditos mensual histórico.
    # Definir variables de tratamiento: canal_contacto y momento (Mañana, Tarde, Noche).
    # Variable objetivo: pago_7d_post_contacto.
    
    # Retornar DataFrame mock/esquema para validación inicial
    return pd.DataFrame()

def build_abt_inferencia(df_clientes, df_creditos):
    """
    Genera la ABT para inferencia/scoring con la foto más reciente de los clientes
    activos con deuda pendiente (mes de corte 2026-03).
    
    Retorna:
        pd.DataFrame: ABT_Inferencia lista para scoring.
    """
    print("Construyendo ABT_Inferencia...")
    # TODO: Filtrar clientes con créditos activos en el último periodo (2026-03)
    # y saldo restante > 0 para realizar la campaña de contacto NBA.
    
    # Retornar DataFrame mock/esquema para validación inicial
    return pd.DataFrame()

def main():
    print("=== Iniciando Fase A: Preparación de Datos ===")
    try:
        df_clientes, df_creditos, df_contactos = load_raw_data()
        print(f"Clientes: {df_clientes.shape}, Créditos: {df_creditos.shape}, Contactos: {df_contactos.shape}")
        
        # 1. Procesar Features y Generar ABTs
        # abt_train = build_abt_train(df_clientes, df_creditos, df_contactos)
        # abt_inferencia = build_abt_inferencia(df_clientes, df_creditos)
        
        # 2. Guardar archivos resultantes
        # os.makedirs(INTERIM_DIR, exist_ok=True)
        # abt_train.to_csv(os.path.join(INTERIM_DIR, "ABT_Train.csv"), index=False)
        # abt_inferencia.to_csv(os.path.join(INTERIM_DIR, "ABT_Inferencia.csv"), index=False)
        # print("Fase A Completada. Master Tables guardadas en data/02_interim/.")
        
    except Exception as e:
        print(f"Error en data_prep.py: {str(e)}")

if __name__ == "__main__":
    main()
