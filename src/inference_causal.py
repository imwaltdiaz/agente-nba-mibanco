"""
Fase B: Inferencia del Modelo Causal
Este script carga el modelo causal entrenado, lee la ABT de Inferencia,
estima los scores (probabilidades de pago y valor neto esperado) para cada
combinación de canal y momento, y genera la tabla de scores para el optimizador.
"""

import os
import pandas as pd
import numpy as np
import joblib

# Configuración de Rutas
INTERIM_DIR = os.path.join(os.path.dirname(__file__), "../data/02_interim")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Costos operativos de contacto por canal (Referenciales)
COSTOS_CANAL = {
    'sms': 0.2,
    'whatsapp': 0.1,
    'llamada': 1.5,
    'campo': 8.0,
    'no_contacto': 0.0
}

def load_inference_data():
    """
    Carga la ABT de Inferencia.
    """
    abt_path = os.path.join(INTERIM_DIR, "ABT_Inferencia.csv")
    if not os.path.exists(abt_path):
        raise FileNotFoundError(f"No se encontró {abt_path}. Ejecuta data_prep.py primero.")
    return pd.read_csv(abt_path)

def load_causal_model():
    """
    Carga el modelo causal guardado.
    """
    model_path = os.path.join(MODELS_DIR, "causal_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró {model_path}. Ejecuta train_causal.py primero.")
    return joblib.load(model_path)

def calculate_expected_value(df_inferencia, model, features):
    """
    Estima la probabilidad de pago y calcula el Valor Neto Esperado (VNE)
    para cada canal y momento posible para todos los clientes en la ABT de Inferencia.
    
    Parámetros:
        df_inferencia (pd.DataFrame): Datos de inferencia de clientes activos.
        model: Modelo causal entrenado.
        features (list): Lista de características del cliente/crédito.
        
    Retorna:
        pd.DataFrame: DataFrame original con columnas de score adicionales (ej: 'whatsapp_manana_vne', etc.).
    """
    print("Estimando probabilidad de pago y Valor Neto Esperado por canal/momento...")
    # TODO: Implementar lógica de scoring y VNE por los otros DS.
    # Para cada cliente se debe simular el impacto financiero de contactar por cada canal/momento:
    # 
    # VNE_canal_momento = P(pago | features, canal, momento) * Min(cuota_mensual, saldo_restante) - Costo_canal
    #
    # El resultado final contendrá una columna de VNE por cada tratamiento posible
    # para ser ingresado directamente al optimizador matemático.
    return df_inferencia

def main():
    print("=== Iniciando Fase B: Inferencia Causal ===")
    try:
        # 1. Cargar datos e inferir
        # df_inferencia = load_inference_data()
        # model = load_causal_model()
        
        # 2. Calcular scores y valor neto esperado
        # features = ['edad', 'score_riesgo', 'prob_default', 'saldo_restante', 'cuota_mensual']
        # df_scored = calculate_expected_value(df_inferencia, model, features)
        
        # 3. Guardar scores para la Fase C
        # scores_path = os.path.join(INTERIM_DIR, "scores_inferencia.csv")
        # df_scored.to_csv(scores_path, index=False)
        # print(f"Scores calculados y guardados en {scores_path}")
        pass
    except Exception as e:
        print(f"Error en inference_causal.py: {str(e)}")

if __name__ == "__main__":
    main()
