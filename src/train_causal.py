"""
Fase B: Entrenamiento de Modelo Causal
Este script entrena el modelo Causal (e.g. EconML/CausalML o un Metalearner customizado)
para estimar el Uplift de los canales de contacto y momentos de día.
Guarda el modelo serializado en models/causal_model.joblib.
"""

import os
import pandas as pd
import numpy as np
import joblib

# Sugerencias de imports para los otros Data Scientists:
# from sklearn.ensemble import RandomForestClassifier
# from econml.metalearners import TLearner

# Configuración de Rutas
INTERIM_DIR = os.path.join(os.path.dirname(__file__), "../data/02_interim")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def load_abt_train():
    """
    Carga la Analytical Base Table de entrenamiento.
    """
    abt_path = os.path.join(INTERIM_DIR, "ABT_Train.csv")
    if not os.path.exists(abt_path):
        raise FileNotFoundError(f"No se encontró {abt_path}. Ejecuta data_prep.py primero.")
    return pd.read_csv(abt_path)

def train_causal_model(df_train, features, treatment_col, outcome_col):
    """
    Entrena el modelo Causal (e.g. T-Learner) usando los features,
    la columna de tratamiento (combinación canal + momento) y la variable objetivo (outcome).
    
    Parámetros:
        df_train (pd.DataFrame): Datos de entrenamiento.
        features (list): Lista de nombres de características.
        treatment_col (str): Nombre de la columna de tratamiento.
        outcome_col (str): Nombre de la columna objetivo.
    """
    print("Entrenando modelo causal...")
    # TODO: Implementar el entrenamiento del modelo causal por los otros DS.
    # Ejemplo:
    # X = df_train[features]
    # T = df_train[treatment_col]  # Canales: sms, whatsapp, llamada, campo (o combinaciones con momentos)
    # y = df_train[outcome_col]    # Pago dentro de 7 días
    #
    # model = TLearner(models=RandomForestClassifier(n_estimators=100, random_state=42))
    # model.fit(y, T, X=X)
    # return model
    return None

def main():
    print("=== Iniciando Fase B: Entrenamiento del Modelo Causal ===")
    try:
        # 1. Cargar datos de entrenamiento
        # df_train = load_abt_train()
        
        # 2. Definir variables
        # features = ['edad', 'score_riesgo', 'prob_default', 'dias_mora_promedio', 'saldo_restante', 'cuota_mensual']
        # treatment_col = 'tratamiento'  # ej. 'sms_Mañana', 'whatsapp_Tarde', etc.
        # outcome_col = 'pago_7d_post_contacto'
        
        # 3. Entrenar el modelo
        # causal_model = train_causal_model(df_train, features, treatment_col, outcome_col)
        
        # 4. Guardar el modelo en models/causal_model.joblib
        # os.makedirs(MODELS_DIR, exist_ok=True)
        # model_path = os.path.join(MODELS_DIR, "causal_model.joblib")
        # joblib.dump(causal_model, model_path)
        # print(f"Modelo causal entrenado y guardado exitosamente en: {model_path}")
        pass
    except Exception as e:
        print(f"Error en train_causal.py: {str(e)}")

if __name__ == "__main__":
    main()
