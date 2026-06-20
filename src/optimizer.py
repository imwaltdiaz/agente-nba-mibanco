"""
Fase C: Optimización de Asignación (NBA)
Este script de PuLP toma los scores y el Valor Neto Esperado calculado en la Fase B,
aplica restricciones operativas y presupuestales, y encuentra la matriz final de asignación
(mejor canal y momento por cliente).
Guarda la salida en data/03_processed/assignment_matrix.csv.
"""

import os
import pandas as pd
import pulp

# Configuración de Rutas
INTERIM_DIR = os.path.join(os.path.dirname(__file__), "../data/02_interim")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/03_processed")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "../logs")

def load_scores():
    """
    Carga los scores de inferencia generados en la Fase B.
    """
    scores_path = os.path.join(INTERIM_DIR, "scores_inferencia.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"No se encontró {scores_path}. Ejecuta inference_causal.py primero.")
    return pd.read_csv(scores_path)

def run_optimization(df_scores, budget, capacities):
    """
    Modela y resuelve el problema de optimización entera usando PuLP.
    
    Parámetros:
        df_scores (pd.DataFrame): DataFrame con la información de los clientes y sus VNE por canal/momento.
        budget (float): Presupuesto máximo asignado para la campaña.
        capacities (dict): Diccionario con capacidades límites de cada canal (ej: {'llamada': 2000, 'campo': 150}).
        
    Retorna:
        pd.DataFrame: Matriz de asignación optimizada por cliente.
    """
    print("Iniciando modelamiento en PuLP...")
    # TODO: Implementar el modelo de optimización lineal entera por los otros DS.
    # 
    # 1. Definir el problema de Maximización
    #    prob = pulp.LpProblem("NBA_Cobranza_Optimization", pulp.LpMaximize)
    #
    # 2. Definir las variables binarias: x_c_ch_m (1 si asignamos cliente c al canal ch en momento m)
    #
    # 3. Definir la Función Objetivo: Maximizar sum(x_c_ch_m * VNE_c_ch_m)
    #
    # 4. Agregar restricciones:
    #    - Unicidad: Cada cliente recibe máximo 1 contacto (o ninguno).
    #    - Presupuesto: sum(x_c_ch_m * Costo_ch) <= Budget
    #    - Capacidades operativas: Límite superior de llamadas, visitas de campo, etc.
    #    - Restricciones por franja horaria.
    #
    # 5. Resolver el problema con pulp.LpSolverDefault.solve() y registrar logs.
    return pd.DataFrame()

def main():
    print("=== Iniciando Fase C: Optimización Matemática ===")
    try:
        # 1. Cargar scores
        # df_scores = load_scores()
        
        # 2. Definir parámetros de negocio iniciales
        budget = 5000.0  # Presupuesto total en soles/dólares
        capacities = {
            'llamada': 2500,  # Capacidad máxima del contact center telefónico
            'campo': 350,     # Capacidad máxima de gestores físicos de cobro en campo
            'whatsapp': 10000, # Límite diario para evitar bloqueos/spam
            'sms': 20000
        }
        
        # 3. Ejecutar optimizador
        # df_assignment = run_optimization(df_scores, budget, capacities)
        
        # 4. Guardar resultados
        # os.makedirs(PROCESSED_DIR, exist_ok=True)
        # assignment_path = os.path.join(PROCESSED_DIR, "assignment_matrix.csv")
        # df_assignment.to_csv(assignment_path, index=False)
        # print(f"Optimización completada exitosamente. Asignación guardada en: {assignment_path}")
        pass
    except Exception as e:
        print(f"Error en optimizer.py: {str(e)}")

if __name__ == "__main__":
    main()
