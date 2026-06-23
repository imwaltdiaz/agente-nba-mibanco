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
import datetime  # <-- Agrega esto
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config

# ---------------------------------------------------------------------------
# CONFIGURACIÓN Y CONSTANTES DE NEGOCIO
# ---------------------------------------------------------------------------
PROCESSED_DIR = config.PROCESSED_DIR

CHANNELS = ["Control", "WhatsApp", "SMS", "Llamada", "Campo"]

# Matriz de Costos Unitarios (Soles)
COSTS = {
    "Control": 0.00,
    "WhatsApp": 1,
    "SMS": 2,
    "Llamada": 3,
    "Campo": 4
}

DAY_NAMES = {
    1: "Lunes",
    2: "Martes",
    3: "Miercoles",
    4: "Jueves",
    5: "Viernes",
    6: "Sabado",
}

CALL_FIELD_DAYS = [1, 2, 3, 4, 5]
DIGITAL_DAYS = [1, 2, 3, 4, 5, 6]
OPERATING_HOURS = list(range(7, 19))  # slots 07:00-08:00 ... 18:00-19:00

# Distribucion historica de contactabilidad por hora. Se usa para priorizar
# horarios con mayor probabilidad operativa; no reemplaza al uplift causal.

HOURLY_WEIGHTS = {
    1: {8: 0.064, 9: 0.105, 10: 0.111, 11: 0.110, 12: 0.101, 13: 0.093, 14: 0.089, 15: 0.090, 16: 0.092, 17: 0.084, 18: 0.061},
    2: {8: 0.069, 9: 0.101, 10: 0.110, 11: 0.106, 12: 0.100, 13: 0.090, 14: 0.086, 15: 0.087, 16: 0.096, 17: 0.093, 18: 0.063},
    3: {8: 0.064, 9: 0.098, 10: 0.108, 11: 0.108, 12: 0.102, 13: 0.092, 14: 0.089, 15: 0.091, 16: 0.093, 17: 0.089, 18: 0.064},
    4: {8: 0.064, 9: 0.097, 10: 0.109, 11: 0.107, 12: 0.100, 13: 0.092, 14: 0.086, 15: 0.092, 16: 0.097, 17: 0.095, 18: 0.061},
    5: {8: 0.066, 9: 0.100, 10: 0.109, 11: 0.108, 12: 0.100, 13: 0.092, 14: 0.089, 15: 0.091, 16: 0.095, 17: 0.090, 18: 0.060},
    6: {8: 0.081, 9: 0.123, 10: 0.144, 11: 0.146, 12: 0.127, 13: 0.091, 14: 0.081, 15: 0.077, 16: 0.069, 17: 0.060, 18: 0.001},
}

def load_scores() -> pd.DataFrame:
    """Carga los scores de inferencia generados en la Fase B.2."""
    scores_path = PROCESSED_DIR / "cate_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"No se encontró {scores_path}. Ejecuta inference_causal.py primero.")
    
    df = pd.read_parquet(scores_path)
    missing_ops = [c.capitalize() for c in config.OPERATIONAL_COLS if c.capitalize() not in df.columns]
    if missing_ops:
        df = enrich_scores_with_operational_data(df)
    print(f"  [OK] CATE Scores cargados: {df.shape[0]:,} clientes.")
    return df


def enrich_scores_with_operational_data(df_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibilidad con scores antiguos: si cate_scores.parquet no trae Region/Zona,
    se enriquecen desde la tabla de clientes cruda.
    """
    clientes_path = config.RAW_DIR / config.RAW_CLIENTES
    if not clientes_path.exists():
        for col in ["Region", "Zona"]:
            if col not in df_scores.columns:
                df_scores[col] = "SinDato"
        return df_scores

    df_clientes = pd.read_csv(clientes_path, usecols=["cliente_id", "region", "zona"])
    df_clientes = df_clientes.rename(
        columns={"cliente_id": "Cliente_ID", "region": "Region", "zona": "Zona"}
    )
    return df_scores.merge(df_clientes, on="Cliente_ID", how="left")

def run_optimization(df_scores: pd.DataFrame, budget: float, capacities: dict) -> pd.DataFrame:
    """
    Modela y resuelve el problema de asignación óptima usando PuLP.
    """
    clientes = df_scores['Cliente_ID'].tolist()
    
    # 1. Mapeo de diccionarios para acceso O(1) en PuLP
    deuda = dict(zip(df_scores['Cliente_ID'], df_scores['Deuda_Expuesta']))
    region = dict(zip(df_scores['Cliente_ID'], df_scores.get('Region', pd.Series('SinDato', index=df_scores.index)).fillna('SinDato')))
    zona = dict(zip(df_scores['Cliente_ID'], df_scores.get('Zona', pd.Series('SinDato', index=df_scores.index)).fillna('SinDato')))
    
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
                    'Region': region[c],
                    'Zona': zona[c],
                    'Deuda_Expuesta': deuda[c],
                    'Canal_Asignado': ch,
                    'Prob_Incremental': uplift[ch][c],
                    'Costo_Incurrido': COSTS[ch],
                    'Valor_Esperado_Neto': VNE[c][ch]
                })
                break
                
    return pd.DataFrame(resultados)


def slot_label(hour: int) -> str:
    return f"{hour:02d}:00-{hour + 1:02d}:00"


def slot_weight(day: int, hour: int) -> float:
    # La hora 7 no estaba en la matriz historica provista; se permite por regla
    # operativa, pero se prioriza por debajo de las horas con historico.
    return HOURLY_WEIGHTS.get(day, {}).get(hour, 0.030)


def ordered_slots(days: list[int]) -> list[tuple[int, int]]:
    slots = [(day, hour) for day in days for hour in OPERATING_HOURS]
    return sorted(slots, key=lambda s: (-slot_weight(s[0], s[1]), s[0], s[1]))


def normalize_region(value) -> str:
    if pd.isna(value):
        return "SinDato"
    return str(value).strip().title() or "SinDato"


def normalize_zone(value) -> str:
    if pd.isna(value):
        return "urbano"
    zone = str(value).strip().lower()
    return zone if zone in {"urbano", "rural"} else "urbano"


def build_slot_capacity(days: list[int], capacity_per_hour: int) -> dict[tuple[int, int], int]:
    return {(day, hour): capacity_per_hour for day in days for hour in OPERATING_HOURS}


def assign_slot_from_capacity(
    remaining: dict[tuple[int, int], int],
    slots: list[tuple[int, int]],
) -> tuple[int | None, int | None]:
    for day, hour in slots:
        if remaining[(day, hour)] > 0:
            remaining[(day, hour)] -= 1
            return day, hour
    return None, None


def schedule_field_visit(
    row: pd.Series,
    field_state: dict,
    capacities: dict,
) -> dict:
    """
    Agenda una visita de campo con restricciones minimas:
      - Solo lunes a viernes.
      - Cada asesor cubre una sola region por dia porque los asesores se crean
        como pools region-dia.
      - Urbano permite hasta 15 visitas por asesor/dia; rural hasta 10.
    """
    region = normalize_region(row.get("Region", "SinDato"))
    zone = normalize_zone(row.get("Zona", "urbano"))
    advisors_by_region = capacities.get(
        "campo_asesores_por_region",
        {"Lima": 8, "Norte": 4, "Sur": 4, "Centro": 4, "Sindato": 2, "SinDato": 2},
    )
    advisors = advisors_by_region.get(region, advisors_by_region.get(region.title(), 2))
    max_visits = (
        capacities.get("campo_visitas_rural_por_asesor_dia", 10)
        if zone == "rural"
        else capacities.get("campo_visitas_urbano_por_asesor_dia", 15)
    )

    for day in CALL_FIELD_DAYS:
        for advisor_idx in range(1, advisors + 1):
            advisor_id = f"CAMPO_{region.upper()}_{advisor_idx:02d}"
            key = (day, region, advisor_id)
            current_load = field_state.setdefault(key, 0)
            if current_load < max_visits:
                field_state[key] += 1
                hour = OPERATING_HOURS[current_load % len(OPERATING_HOURS)]
                return {
                    "Dia_Semana": DAY_NAMES[day],
                    "Hora_Inicio": f"{hour:02d}:00",
                    "Hora_Fin": f"{hour + 1:02d}:00",
                    "Slot_Asignado": f"{DAY_NAMES[day]} {slot_label(hour)}",
                    "Asesor_Asignado": advisor_id,
                    "Estado_Agenda": "AGENDADO",
                    "Restriccion_Aplicada": f"Campo {region}/{zone}: max {max_visits} visitas asesor-dia",
                }

    return {
        "Dia_Semana": None,
        "Hora_Inicio": None,
        "Hora_Fin": None,
        "Slot_Asignado": None,
        "Asesor_Asignado": None,
        "Estado_Agenda": "SIN_CAPACIDAD",
        "Restriccion_Aplicada": f"Sin capacidad campo en region {region}",
    }


def schedule_assignments(df_assignment: pd.DataFrame, capacities: dict) -> pd.DataFrame:
    """
    Segunda etapa del motor NBA: agenda dia/hora luego de elegir el canal.

    Reglas:
      - Llamada y Campo: lunes a viernes, 07:00-19:00.
      - WhatsApp y SMS: lunes a sabado, 07:00-19:00.
      - Sabado solo canal digital.
      - Domingo no se agenda.
      - Campo respeta region y limite de visitas por asesor/dia.
    """
    df = df_assignment.copy()
    df["_priority"] = df["Valor_Esperado_Neto"].rank(method="first", ascending=False)
    df = df.sort_values(["Canal_Asignado", "_priority"])

    call_remaining = build_slot_capacity(
        CALL_FIELD_DAYS,
        capacities.get("llamada_por_hora", 150),
    )
    digital_remaining = build_slot_capacity(
        DIGITAL_DAYS,
        capacities.get("digital_por_hora", 2000),
    )
    call_slots = ordered_slots(CALL_FIELD_DAYS)
    digital_slots = ordered_slots(DIGITAL_DAYS)
    field_state = {}

    scheduled_rows = []
    for _, row in df.iterrows():
        channel = row["Canal_Asignado"]
        schedule = {
            "Dia_Semana": None,
            "Hora_Inicio": None,
            "Hora_Fin": None,
            "Slot_Asignado": None,
            "Asesor_Asignado": None,
            "Estado_Agenda": "NO_REQUIERE_CONTACTO" if channel == "Control" else "SIN_AGENDAR",
            "Restriccion_Aplicada": "Control: no contactar" if channel == "Control" else None,
        }

        if channel == "Llamada":
            day, hour = assign_slot_from_capacity(call_remaining, call_slots)
            if day is not None:
                schedule.update({
                    "Dia_Semana": DAY_NAMES[day],
                    "Hora_Inicio": f"{hour:02d}:00",
                    "Hora_Fin": f"{hour + 1:02d}:00",
                    "Slot_Asignado": f"{DAY_NAMES[day]} {slot_label(hour)}",
                    "Asesor_Asignado": "CALL_CENTER_POOL",
                    "Estado_Agenda": "AGENDADO",
                    "Restriccion_Aplicada": "Llamada L-V 07:00-19:00",
                })
            else:
                schedule["Estado_Agenda"] = "SIN_CAPACIDAD"
                schedule["Restriccion_Aplicada"] = "Sin capacidad call center"

        elif channel in {"WhatsApp", "SMS"}:
            day, hour = assign_slot_from_capacity(digital_remaining, digital_slots)
            if day is not None:
                schedule.update({
                    "Dia_Semana": DAY_NAMES[day],
                    "Hora_Inicio": f"{hour:02d}:00",
                    "Hora_Fin": f"{hour + 1:02d}:00",
                    "Slot_Asignado": f"{DAY_NAMES[day]} {slot_label(hour)}",
                    "Asesor_Asignado": "DIGITAL_AUTOMATION",
                    "Estado_Agenda": "AGENDADO",
                    "Restriccion_Aplicada": "Digital L-S 07:00-19:00; domingo bloqueado",
                })
            else:
                schedule["Estado_Agenda"] = "SIN_CAPACIDAD"
                schedule["Restriccion_Aplicada"] = "Sin capacidad digital"

        elif channel == "Campo":
            schedule = schedule_field_visit(row, field_state, capacities)

        scheduled_rows.append({**row.drop(labels=["_priority"]).to_dict(), **schedule})

    output = pd.DataFrame(scheduled_rows)
    return output.sort_values(
        ["Estado_Agenda", "Dia_Semana", "Hora_Inicio", "Valor_Esperado_Neto"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def print_table(title: str, df: pd.DataFrame | pd.Series, max_rows: int | None = None) -> None:
    """Imprime tablas compactas de KPIs sin romper si vienen vacias."""
    print(f"\n{title}")
    print("-" * len(title))
    if df is None or len(df) == 0:
        print("  Sin datos.")
        return
    if max_rows is not None:
        df = df.head(max_rows)
    print(df.to_string())


def print_channel_assignment_kpis(df_assignment: pd.DataFrame) -> None:
    """
    Vista macro de la optimizacion de canales antes de la agenda horaria.
    """
    total = len(df_assignment)
    channel_summary = (
        df_assignment.groupby("Canal_Asignado")
        .agg(
            Clientes=("Cliente_ID", "count"),
            Deuda_Total=("Deuda_Expuesta", "sum"),
            Deuda_Promedio=("Deuda_Expuesta", "mean"),
            Uplift_Promedio=("Prob_Incremental", "mean"),
            Costo_Total=("Costo_Incurrido", "sum"),
            VNE_Total=("Valor_Esperado_Neto", "sum"),
            VNE_Promedio=("Valor_Esperado_Neto", "mean"),
        )
        .sort_values("Clientes", ascending=False)
    )
    channel_summary["Pct_Clientes"] = (channel_summary["Clientes"] / total * 100).round(2)
    channel_summary = channel_summary[
        [
            "Clientes", "Pct_Clientes", "Deuda_Total", "Deuda_Promedio",
            "Uplift_Promedio", "Costo_Total", "VNE_Total", "VNE_Promedio",
        ]
    ].round(3)

    print("\n" + "-" * 70)
    print(" KPIs MACRO - ASIGNACION DE CANALES")
    print("-" * 70)
    print(f" Clientes evaluados       : {total:,}")
    print(f" Clientes contactados     : {(df_assignment['Canal_Asignado'] != 'Control').sum():,}")
    print(f" Clientes en control      : {(df_assignment['Canal_Asignado'] == 'Control').sum():,}")
    print(f" Deuda expuesta total     : S/ {df_assignment['Deuda_Expuesta'].sum():,.2f}")
    print(f" Costo total asignado     : S/ {df_assignment['Costo_Incurrido'].sum():,.2f}")
    print(f" VNE total asignado       : S/ {df_assignment['Valor_Esperado_Neto'].sum():,.2f}")
    print(f" Uplift promedio cartera  : {df_assignment['Prob_Incremental'].mean():.4f}")
    print_table("Resumen por canal", channel_summary)


def print_schedule_kpis(df_scheduled: pd.DataFrame) -> None:
    """
    Vista macro de la agenda final: volumen por dia/hora/canal y KPIs de campo.
    """
    scheduled = df_scheduled[df_scheduled["Estado_Agenda"] == "AGENDADO"].copy()
    contacted = df_scheduled[df_scheduled["Canal_Asignado"] != "Control"].copy()
    field = scheduled[scheduled["Canal_Asignado"] == "Campo"].copy()
    calls = scheduled[scheduled["Canal_Asignado"] == "Llamada"].copy()
    digital = scheduled[scheduled["Canal_Asignado"].isin(["WhatsApp", "SMS"])].copy()

    print("\n" + "-" * 70)
    print(" KPIs MACRO - AGENDA OPTIMIZADA CON HORARIOS")
    print("-" * 70)
    print(f" Contactos a ejecutar       : {len(contacted):,}")
    print(f" Contactos agendados        : {len(scheduled):,}")
    print(f" Sin capacidad              : {(df_scheduled['Estado_Agenda'] == 'SIN_CAPACIDAD').sum():,}")
    print(f" No requiere contacto       : {(df_scheduled['Estado_Agenda'] == 'NO_REQUIERE_CONTACTO').sum():,}")
    if len(contacted) > 0:
        print(f" Cobertura agenda/contacto  : {len(scheduled) / len(contacted) * 100:.2f}%")
    print(f" Promedio VNE agendado      : S/ {scheduled['Valor_Esperado_Neto'].mean():,.2f}")
    print(f" Promedio deuda agendada    : S/ {scheduled['Deuda_Expuesta'].mean():,.2f}")

    status_counts = df_scheduled["Estado_Agenda"].value_counts().rename("Clientes")
    print_table("Estado de agenda", status_counts)

    if len(scheduled) > 0:
        by_day = pd.crosstab(scheduled["Dia_Semana"], scheduled["Canal_Asignado"])
        by_day["Total"] = by_day.sum(axis=1)
        print_table("Contactos agendados por dia y canal", by_day)

        by_hour = pd.crosstab(scheduled["Hora_Inicio"], scheduled["Canal_Asignado"])
        by_hour["Total"] = by_hour.sum(axis=1)
        print_table("Contactos agendados por hora y canal", by_hour)

        by_slot = (
            scheduled.groupby(["Dia_Semana", "Hora_Inicio"])
            .agg(Contactos=("Cliente_ID", "count"), VNE_Total=("Valor_Esperado_Neto", "sum"))
            .sort_values(["Dia_Semana", "Hora_Inicio"])
            .round(2)
        )
        print_table("Carga por dia-hora", by_slot, max_rows=80)

    if len(calls) > 0:
        call_day = calls.groupby("Dia_Semana").agg(
            Llamadas=("Cliente_ID", "count"),
            VNE_Total=("Valor_Esperado_Neto", "sum"),
            VNE_Promedio=("Valor_Esperado_Neto", "mean"),
        ).round(2)
        print_table("Call center por dia", call_day)
        print(f"\n Promedio llamadas por dia : {len(calls) / calls['Dia_Semana'].nunique():.1f}")
        print(f" Pico llamadas por hora    : {calls.groupby(['Dia_Semana', 'Hora_Inicio']).size().max():,}")

    if len(digital) > 0:
        digital_summary = digital.groupby(["Dia_Semana", "Canal_Asignado"]).agg(
            Contactos=("Cliente_ID", "count"),
            VNE_Total=("Valor_Esperado_Neto", "sum"),
        ).round(2)
        print_table("Canales digitales por dia", digital_summary)
        print(f"\n Promedio digitales por dia: {len(digital) / digital['Dia_Semana'].nunique():.1f}")
        print(f" Pico digitales por hora   : {digital.groupby(['Dia_Semana', 'Hora_Inicio']).size().max():,}")

    if len(field) > 0:
        field_region = pd.crosstab(
            [field["Dia_Semana"], field["Region"]],
            field["Zona"],
            margins=True,
            margins_name="Total",
        )
        print_table("Visitas de campo por dia, region y zona", field_region)

        advisor_load = field.groupby(["Dia_Semana", "Asesor_Asignado"]).agg(
            Visitas=("Cliente_ID", "count"),
            VNE_Total=("Valor_Esperado_Neto", "sum"),
        ).sort_values("Visitas", ascending=False).round(2)
        print_table("Carga de asesores de campo", advisor_load, max_rows=30)
        print(f"\n Asesores de campo usados  : {field['Asesor_Asignado'].nunique():,}")
        print(f" Promedio visitas/asesor-dia: {advisor_load['Visitas'].mean():.2f}")
        print(f" Max visitas/asesor-dia     : {advisor_load['Visitas'].max():,}")
        print(f" Promedio visitas por dia   : {len(field) / field['Dia_Semana'].nunique():.1f}")
def calcular_baseline_aleatorio(df_scores, df_assignment=None, budget=5000.0):
    """
    Simula el Valor Neto Esperado de una estrategia tradicional o aleatoria:
    Se asigna a los clientes un canal al azar (probabilidad uniforme entre 
    WhatsApp, SMS, Llamada y Campo) hasta agotar el presupuesto diario.
    """
    canales_activos = ["WhatsApp", "SMS", "Llamada", "Campo"]
    
    # 1. Costo promedio de elegir un canal a ciegas
    # (0.10 + 0.20 + 1.50 + 8.00) / 4 = S/ 2.45 en promedio por contacto
    costo_promedio_azar = sum(COSTS[c] for c in canales_activos) / len(canales_activos)
    
    # 2. ¿A cuántas personas podemos contactar antes de quedarnos sin presupuesto?
    # 5000 / 2.45 = ~2,040 clientes contactados
    n_contactos_posibles = budget / costo_promedio_azar
    
    # 3. Asumiendo una distribución uniforme (25% del presupuesto a cada canal)
    contactos_por_canal = n_contactos_posibles / len(canales_activos)
    
    deuda_promedio = df_scores['Deuda_Expuesta'].mean()
    vne_aleatorio_total = 0.0
    
    for canal in canales_activos:
        # Usamos el uplift promedio general porque la asignación es al azar
        uplift_promedio = df_scores[f'Uplift_{canal}'].mean()
        
        # VNE de aplicar este canal a un cliente promedio
        vne_promedio_canal = (uplift_promedio * deuda_promedio) - COSTS[canal]
        
        # Sumamos el valor generado por la cantidad de personas contactadas
        vne_aleatorio_total += (vne_promedio_canal * contactos_por_canal)
        
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
            'sms': 20000,
            'llamada_por_hora': 150,
            'digital_por_hora': 2000,
            'campo_asesores_por_region': {
                'Lima': 8,
                'Norte': 4,
                'Sur': 4,
                'Centro': 4,
                'SinDato': 2,
            },
            'campo_visitas_urbano_por_asesor_dia': 15,
            'campo_visitas_rural_por_asesor_dia': 10,
        }
        
        # 3. Ejecutar Agente
        print("\n[2/4] Ejecutando motor de optimización prescriptiva...")
        df_assignment = run_optimization(df_scores, budget, capacities)
        print_channel_assignment_kpis(df_assignment)
        
        # 4. Agendar y exportar
        print("\n[3/4] Agendando contactos por dia/hora segun restricciones operativas...")
        df_scheduled = schedule_assignments(df_assignment, capacities)
        print_schedule_kpis(df_scheduled)

        print("\n[4/4] Guardando matriz de asignación final...")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        # Generar sufijo único con la hora exacta
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        version = "v2_agenda"

        channel_assignment_path = PROCESSED_DIR / f"asignacion_canales_{version}_{timestamp}.csv"
        assignment_path = PROCESSED_DIR / f"asignacion_optimizada_{version}_{timestamp}.csv"
        
        df_assignment.to_csv(channel_assignment_path, index=False)
        df_scheduled.to_csv(assignment_path, index=False)
        
        # --- RESUMEN DE NEGOCIO PARA EL JURADO ---
        costo_total = df_scheduled['Costo_Incurrido'].sum()
        retorno_incremental = df_scheduled['Valor_Esperado_Neto'].sum()
        
        print(f"\n{'-'*60}")
        print(" RESUMEN DE EJECUCIÓN DIARIA (IMPACTO EN NEGOCIO)")
        print(f"{'-'*60}")
        print(" Distribución de Canales:")
        print(df_scheduled['Canal_Asignado'].value_counts().to_string())
        print(f"\n Restricciones Aplicadas:")
        print(f"  - Presupuesto Max: S/ {budget:,.2f} | Consumido: S/ {costo_total:,.2f}")
        print(f"  - Capacidad Campo: {capacities['campo']} | Asignados: {(df_scheduled['Canal_Asignado'] == 'Campo').sum()}")
        print(f"  - Campo urbano: max {capacities['campo_visitas_urbano_por_asesor_dia']} visitas asesor/dia")
        print(f"  - Campo rural:  max {capacities['campo_visitas_rural_por_asesor_dia']} visitas asesor/dia")
        print(f"  - Agenda sin capacidad: {(df_scheduled['Estado_Agenda'] == 'SIN_CAPACIDAD').sum()}")
        print(f"\n 💰 Retorno Incremental Esperado (ROI): S/ {retorno_incremental:,.2f}")
        print(f" 💾 Canales guardados en: {channel_assignment_path}")
        print(f" 💾 Agenda final guardada en: {assignment_path}")
        print("="*60 + "\n")
        

        # -- evaluacion
        # --- MONITOREO DE MÉTRICAS COMPLETO (KPIs DEL AGENTE) ---
        total_clientes = len(df_scheduled)
        costo_total = df_scheduled['Costo_Incurrido'].sum()
        vne_total = df_scheduled['Valor_Esperado_Neto'].sum()
        
        # --- NUEVO CÁLCULO DE COMPARACIÓN ---
        vne_azar = calcular_baseline_aleatorio(df_scores, df_scheduled)
        valor_agregado_ia = vne_total - vne_azar

        # 1. KPIs Operativos
        eficiencia_presupuesto = (costo_total / budget) * 100
        costo_medio_contacto = costo_total / total_clientes
        tasa_autolimpiado_control = (df_scheduled['Canal_Asignado'] == 'Control').mean() * 100
        
        # 2. KPIs Financieros
        # ROI multiplicador (cuántas veces recuperas la inversión operativa)
        roi_multiplicador = vne_total / costo_total if costo_total > 0 else 0.0
        
        # Cuota mensual promedio de los clientes impactados (Ticket Promedio)
        ticket_promedio_deuda = df_scheduled[df_scheduled['Canal_Asignado'] != 'Control']['Deuda_Expuesta'].mean()

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
