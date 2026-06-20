"""
optimizer.py - Fase C: Optimizador Matemático del Agente Orquestador NBA
==========================================================================
Responsabilidades:
  1. Carga los scores CATE (Uplift) generados en la Fase B.2.
  2. Construye la matriz de Valor Neto Esperado (VNE) por cada cliente y canal.
     Fórmula: VNE = (Probabilidad_Incremental * Deuda_Expuesta) - Costo_Canal
  3. Modela el problema de optimización en PuLP (Algoritmo de la Mochila/Asignación).
  4. Resuelve garantizando restricciones de capacidad operativa y presupuesto.
  5. Exporta la matriz final de asignación.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# CONFIGURACIÓN Y CONSTANTES DE NEGOCIO
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "03_processed"

CHANNELS = ["Control", "WhatsApp", "SMS", "Llamada", "Campo"]

# Matriz de Costos Unitarios (Soles)
COSTS = {
    "Control": 0.00,
    "WhatsApp": 0.10,
    "SMS": 0.20,
    "Llamada": 1.50,
    "Campo": 8.00
}

def load_scores() -> pd.DataFrame:
    """Carga los scores de inferencia generados en la Fase B.2."""
    scores_path = PROCESSED_DIR / "cate_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"No se encontró {scores_path}. Ejecuta inference_causal.py primero.")
    
    df = pd.read_parquet(scores_path)
    print(f"  [OK] CATE Scores cargados: {df.shape[0]:,} clientes.")
    return df

def run_optimization(df_scores: pd.DataFrame, budget: float, capacities: dict) -> pd.DataFrame:
    """
    Modela y resuelve el problema de asignación óptima usando PuLP.
    """
    clientes = df_scores['Cliente_ID'].tolist()
    
    # 1. Mapeo de diccionarios para acceso O(1) en PuLP
    deuda = dict(zip(df_scores['Cliente_ID'], df_scores['Deuda_Expuesta']))
    
    uplift = {
        'WhatsApp': dict(zip(df_scores['Cliente_ID'], df_scores['Uplift_WhatsApp'])),
        'SMS': dict(zip(df_scores['Cliente_ID'], df_scores['Uplift_SMS'])),
        'Llamada': dict(zip(df_scores['Cliente_ID'], df_scores['Uplift_Llamada'])),
        'Campo': dict(zip(df_scores['Cliente_ID'], df_scores['Uplift_Campo'])),
        'Control': {c: 0.0 for c in clientes} # El control tiene 0 uplift incremental
    }
    
    # 2. Calcular la Matriz de Valor Neto Esperado (VNE)
    print("  Calculando Matriz de Valor Neto Esperado (VNE)...")
    VNE = {}
    for c in clientes:
        VNE[c] = {}
        for ch in CHANNELS:
            u = uplift[ch][c]
            d = deuda[c]
            costo = COSTS[ch]
            # VNE: Retorno esperado incremental restando la inversión operativa
            VNE[c][ch] = (u * d) - costo

    # 3. Definir el problema en PuLP
    print("  Construyendo modelo MILP en PuLP...")
    prob = pulp.LpProblem("Agente_NBA_Cobranza", pulp.LpMaximize)

    # Variables de Decisión: Binarias (1 si asignamos el canal 'ch' al cliente 'c', 0 si no)
    x = pulp.LpVariable.dicts("Asignacion", (clientes, CHANNELS), cat='Binary')

    # Función Objetivo: Maximizar el Valor Neto Esperado total de la cartera
    prob += pulp.lpSum(VNE[c][ch] * x[c][ch] for c in clientes for ch in CHANNELS), "Maximizar_VNE"

    # 4. Restricciones Duras
    # A. Unicidad: Cada cliente recibe exactamente 1 tratamiento (Control significa "no tocar")
    for c in clientes:
        prob += pulp.lpSum(x[c][ch] for ch in CHANNELS) == 1, f"Unicidad_{c}"

    # B. Presupuesto Total Diario
    prob += pulp.lpSum(COSTS[ch] * x[c][ch] for c in clientes for ch in CHANNELS) <= budget, "Presupuesto_Maximo"

    # C. Capacidades Físicas / Operativas
    prob += pulp.lpSum(x[c]['Llamada'] for c in clientes) <= capacities.get('llamada', 999999), "Capacidad_Llamadas"
    prob += pulp.lpSum(x[c]['Campo'] for c in clientes) <= capacities.get('campo', 999999), "Capacidad_Campo"
    prob += pulp.lpSum(x[c]['WhatsApp'] for c in clientes) <= capacities.get('whatsapp', 999999), "Capacidad_WhatsApp"
    prob += pulp.lpSum(x[c]['SMS'] for c in clientes) <= capacities.get('sms', 999999), "Capacidad_SMS"

    # 5. Resolver el problema
    print("  Resolviendo optimización (CBC Solver)...")
    # prob.solve(pulp.PULP_CBC_CMD(msg=0)) # Ejecución silenciosa : lo comento por si tarda mucho mas de 10 min, queremos un resultado rapido al 99.9%, no al 100%
    # Línea optimizada para Hackathon (Resuelve en menos de 15 segundos)
    # Reemplazo con la nomenclatura correcta de la librería
    prob.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=0.01, threads=4))
    status = pulp.LpStatus[prob.status]
    
    if status != 'Optimal':
        print(f"  [WARN] El optimizador no encontró una solución óptima. Estado: {status}")

    # 6. Parsear Resultados
    resultados = []
    for c in clientes:
        for ch in CHANNELS:
            if pulp.value(x[c][ch]) == 1.0:
                resultados.append({
                    'Cliente_ID': c,
                    'Deuda_Expuesta': deuda[c],
                    'Canal_Asignado': ch,
                    'Prob_Incremental': uplift[ch][c],
                    'Costo_Incurrido': COSTS[ch],
                    'Valor_Esperado_Neto': VNE[c][ch]
                })
                break
                
    return pd.DataFrame(resultados)

def calcular_baseline_aleatorio(df_scores, df_assignment):
    """
    Simula el Valor Neto Esperado si la misma cantidad de recursos 
    se asignara a clientes completamente al azar.
    """
    # Tomamos la distribución exacta que decidió el optimizador (ej. 10000 WA, 230 Campo)
    distribucion = df_assignment['Canal_Asignado'].value_counts().to_dict()
    
    deuda_promedio = df_scores['Deuda_Expuesta'].mean()
    vne_aleatorio_total = 0.0
    
    for canal, cantidad in distribucion.items():
        if canal == 'Control':
            continue
            
        # Efecto promedio de aplicar este canal a cualquier persona al azar
        uplift_promedio_canal = df_scores[f'Uplift_{canal}'].mean()
        costo_canal = COSTS[canal]
        
        # VNE de un cliente promedio
        vne_promedio = (uplift_promedio_canal * deuda_promedio) - costo_canal
        
        # Multiplicamos por la cantidad de personas contactadas por ese canal
        vne_aleatorio_total += (vne_promedio * cantidad)
        
    return vne_aleatorio_total

def main():
    print("\n" + "="*60)
    print(" AGENTE ORQUESTADOR NBA - FASE C: OPTIMIZACIÓN MATEMÁTICA")
    print("="*60)

    try:
        # 1. Cargar datos
        print("\n[1/3] Ingiriendo scores de propensión causal...")
        df_scores = load_scores()
        
        # 2. Definir parámetros (Podrían venir de un config file en producción)
        budget = 5000.0  
        capacities = {
            'llamada': 2000, 
            'campo': 500,    
            'whatsapp': 10000, 
            'sms': 20000
        }
        
        # 3. Ejecutar Agente
        print("\n[2/3] Ejecutando motor de optimización prescriptiva...")
        df_assignment = run_optimization(df_scores, budget, capacities)
        
        # 4. Exportar y Resumir
        print("\n[3/3] Guardando matriz de asignación final...")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        assignment_path = PROCESSED_DIR / "asignacion_optimizada.csv"
        df_assignment.to_csv(assignment_path, index=False)
        
        # --- RESUMEN DE NEGOCIO PARA EL JURADO ---
        costo_total = df_assignment['Costo_Incurrido'].sum()
        retorno_incremental = df_assignment['Valor_Esperado_Neto'].sum()
        
        print(f"\n{'-'*60}")
        print(" RESUMEN DE EJECUCIÓN DIARIA (IMPACTO EN NEGOCIO)")
        print(f"{'-'*60}")
        print(" Distribución de Canales:")
        print(df_assignment['Canal_Asignado'].value_counts().to_string())
        print(f"\n Restricciones Aplicadas:")
        print(f"  - Presupuesto Max: S/ {budget:,.2f} | Consumido: S/ {costo_total:,.2f}")
        print(f"  - Capacidad Campo: {capacities['campo']} | Asignados: {(df_assignment['Canal_Asignado'] == 'Campo').sum()}")
        print(f"\n 💰 Retorno Incremental Esperado (ROI): S/ {retorno_incremental:,.2f}")
        print(f" 💾 Matriz guardada en: {assignment_path}")
        print("="*60 + "\n")
        

        # -- evaluacion
        # --- MONITOREO DE MÉTRICAS COMPLETO (KPIs DEL AGENTE) ---
        total_clientes = len(df_assignment)
        costo_total = df_assignment['Costo_Incurrido'].sum()
        vne_total = df_assignment['Valor_Esperado_Neto'].sum()
        
        # --- NUEVO CÁLCULO DE COMPARACIÓN ---
        vne_azar = calcular_baseline_aleatorio(df_scores, df_assignment)
        valor_agregado_ia = vne_total - vne_azar

        # 1. KPIs Operativos
        eficiencia_presupuesto = (costo_total / budget) * 100
        costo_medio_contacto = costo_total / total_clientes
        tasa_autolimpiado_control = (df_assignment['Canal_Asignado'] == 'Control').mean() * 100
        
        # 2. KPIs Financieros
        # ROI multiplicador (cuántas veces recuperas la inversión operativa)
        roi_multiplicador = vne_total / costo_total if costo_total > 0 else 0.0
        
        # Cuota mensual promedio de los clientes impactados (Ticket Promedio)
        ticket_promedio_deuda = df_assignment[df_assignment['Canal_Asignado'] != 'Control']['Deuda_Expuesta'].mean()

        print(f"\n{'-'*60}")
        print("   KPIs DE EVALUACIÓN DEL AGENTE ORQUESTADOR NBA")
        print(f"{'-'*60}")
        print(f" Nivel 1: Eficiencia Operativa y Presupuestal")
        print(f"  - Eficiencia del Gasto Diario : {eficiencia_presupuesto:.2f}% (S/ {costo_total:,.2f} consumidos)")
        print(f"  - Costo de Contacto Promedio  : S/ {costo_medio_contacto:.3f} por cliente")
        print(f"  - Tasa de Control (Sleeping Dogs): {tasa_autolimpiado_control:.2f}% de la cartera protegida")
        
        print(f"\n Nivel 2: Métricas de Valor de Negocio")
        print(f"  - Ticket Promedio de Deuda     : S/ {ticket_promedio_deuda:,.2f} por cuota")
        print(f"  - Retorno Incremental Neto (VNE): S/ {vne_total:,.2f}")
        print(f"  - ROI Multiplicador de Campaña : {roi_multiplicador:.2f}x")
        print(f"    (Por cada Sol invertido se generan S/ {roi_multiplicador:.2f} en valor esperado)")
        print(f"{'-'*60}\n")

        print(f"\n Nivel 3: El Valor Real de la Inteligencia Artificial (A/B Test Simulado)")
        print(f"  - VNE con Asignación al Azar   : S/ {vne_azar:,.2f}")
        print(f"  - VNE con Optimizador (IA)     : S/ {vne_total:,.2f}")
        print(f"  - 🚀 Valor Neto Agregado por IA: S/ {valor_agregado_ia:,.2f}")
        print(f"    (Esto es dinero puro que Mibanco perdería sin este algoritmo)")

    except Exception as e:
        print(f"\n[ERROR] Error inesperado en optimizer.py: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    main()