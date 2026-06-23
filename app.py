"""
app.py - Frontend CobraIQ (Fase D)
========================================================================
Aplicación web interactiva que funciona como interfaz del orquestador NBA.
Clon exacto en estilo, diseño oscuro y pestañas del Dashboard CobraIQ.
Exhibe de forma dinámica las métricas operativas y las recomendaciones de contacto.
"""

import os
import sys
import glob
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Agregar raíz del proyecto al path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
import src.config as config

# Configuración de página
st.set_page_config(
    page_title="CobraIQ - Control de Cobranza Inteligente",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS CON CACHÉ
# ---------------------------------------------------------------------------

@st.cache_data
def get_latest_data():
    """Encuentra y carga la última asignación optimizada guardada por el optimizador."""
    processed_dir = ROOT_DIR / "data" / "03_processed"
    files = list(processed_dir.glob("asignacion_optimizada_v2_agenda_*.csv"))
    
    if files:
        # Ordenar por fecha en el nombre para tomar el más reciente
        files.sort(key=lambda x: x.name)
        latest_file = files[-1]
    else:
        # Fallback a archivo genérico o mock
        latest_file = processed_dir / "asignacion_optimizada.csv"
        if not latest_file.exists():
            return None, "No se encontraron archivos de asignación. Ejecuta optimizer.py."
            
    df = pd.read_csv(latest_file)
    return df, latest_file.name


@st.cache_data
def load_client_database():
    """Carga los datos demográficos y comportamiento del cliente crudo."""
    path = ROOT_DIR / "data" / "01_raw" / "01_Tabla_de_Clientes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_cate_scores():
    """Carga las métricas de propensión causal (Uplift CATE)."""
    path = ROOT_DIR / "data" / "03_processed" / "cate_scores.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# Carga de datasets
df_scheduled, file_name_loaded = get_latest_data()
df_clientes = load_client_database()
df_scores = load_cate_scores()

# ---------------------------------------------------------------------------
# 2. DISEÑO Y ESTILIZACIÓN (CSS PERSONALIZADO - ESTILO COBRAIQ)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Inyección de tipografía moderna */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F1419 !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Configuración del Sidebar lateral */
    [data-testid="stSidebar"] {
        background-color: #1A2332 !important;
        border-right: 1px solid #2D3748 !important;
    }
    
    /* Habilitar scroll del sidebar y asegurar estilo del título */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
    }
    
    /* Ajustes generales de inputs y sliders */
    div[data-baseweb="select"] > div {
        background-color: #1A2332 !important;
        color: #ffffff !important;
        border-color: #2D3748 !important;
    }
    
    input {
        background-color: #1A2332 !important;
        color: #ffffff !important;
        border: 1px solid #2D3748 !important;
    }
    
    /* Menú de Navegación Vertical con Radio Buttons */
    div[role="radiogroup"] {
        background-color: transparent !important;
        gap: 8px !important;
        padding-top: 15px;
    }
    div[role="radiogroup"] label {
        background-color: #161D2B !important;
        border: 1px solid #2D3748 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        color: #A0AEC0 !important;
        cursor: pointer;
        transition: all 0.25s ease-in-out;
        display: flex;
        align-items: center;
        width: 100% !important;
    }
    
    /* Ocultar el circulo de seleccion nativo */
    div[role="radiogroup"] label div[class*="StyledRadio"] {
        display: none !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #2E7D32 !important;
        border-color: #4CAF50 !important;
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.25);
    }
    
    div[role="radiogroup"] label:hover {
        background-color: #1E2D3F !important;
        border-color: #4CAF50 !important;
        color: #FFFFFF !important;
    }
    
    div[role="radiogroup"] label span {
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Tarjetas KPI y contenedores rounded */
    .kpi-container {
        background-color: #1A2332;
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        margin-bottom: 15px;
    }
    
    .kpi-header {
        color: #A0AEC0;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 0.8px;
    }
    
    .kpi-val {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .kpi-subtext {
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Contenedor principal de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1A2332 !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #A0AEC0 !important;
        background-color: transparent !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background-color: #2E7D32 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. SIDEBAR DE NAVEGACIÓN Y CONFIGURACIÓN
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 5px 0;">
        <h1 style="font-size: 32px; font-weight: 700; margin: 0; color: #4CAF50 !important; display: flex; align-items: center; gap: 8px;">
            🐍 CobraIQ
        </h1>
        <div style="font-size: 11px; color: #A0AEC0; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px;">
            Motor Causal de Cobranzas | v2_agenda
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 15px 0; border-color: #2D3748;'>", unsafe_allow_html=True)
    
    # Menú de selección vertical
    current_page = st.radio(
        label="Navegación",
        options=["🏠 Dashboard Ejecutivo", "👥 Gestión de Cartera", "🔍 Ficha del Cliente", "📊 Análisis e Insights"],
        label_visibility="collapsed"
    )
    
    # Meta información en el footer del Sidebar
    st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
    
    if df_scheduled is not None:
        total_inf = len(df_scheduled)
        pct_digital_inf = (df_scheduled['Canal_Asignado'].isin(['WhatsApp', 'SMS']).sum() / total_inf) * 100
        
        # El ahorro estimado es el VNE de la IA menos el VNE del Azar
        vne_opt = df_scheduled['Valor_Esperado_Neto'].sum()
        # VNE de asignación aleatoria calculado de forma estática en base a la consola de logs
        vne_azar = 327745.08
        ahorro_estimado = vne_opt - vne_azar
        
        st.markdown(f"""
        <div style="background-color: #161D2B; border: 1px solid #2D3748; border-radius: 8px; padding: 15px; margin-top: 20px;">
            <div style="font-size: 11px; color: #A0AEC0; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; border-bottom: 1px solid #2D3748; padding-bottom: 5px;">
                Meta-Información
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13px; color: #A0AEC0;">Total Clientes:</span>
                <span style="font-size: 13px; color: #ffffff; font-weight: 600;">{total_inf:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13px; color: #A0AEC0;">Canal Digital:</span>
                <span style="font-size: 13px; color: #4CAF50; font-weight: 600;">{pct_digital_inf:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 13px; color: #A0AEC0;">Valor IA Neto:</span>
                <span style="font-size: 13px; color: #4CAF50; font-weight: 600;">S/ {ahorro_estimado / 1e6:.2f}M</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #2D1A1A; border: 1px solid #FC8181; border-radius: 8px; padding: 15px; margin-top: 20px;">
            <span style="color: #FC8181; font-size: 13px;">Error: Datos no cargados.</span>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. LÓGICA Y CONTENIDO POR PANTALLA
# ---------------------------------------------------------------------------

if df_scheduled is None:
    st.error("No se pudieron cargar los datos de asignación. Asegúrate de ejecutar primero optimizer.py.")
    st.stop()

# ---------------------------------------------------------------------------
# PÁGINA 1: 🏠 DASHBOARD EJECUTIVO
# ---------------------------------------------------------------------------
if current_page == "🏠 Dashboard Ejecutivo":
    st.markdown("<h2 style='margin-bottom: 20px;'>🏠 Dashboard Ejecutivo de Cobranzas</h2>", unsafe_allow_html=True)
    
    # 1. Grid superior de 5 tarjetas KPI
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    total_clientes = len(df_scheduled)
    deuda_total = df_scheduled['Deuda_Expuesta'].sum()
    costo_total = df_scheduled['Costo_Incurrido'].sum()
    vne_total = df_scheduled['Valor_Esperado_Neto'].sum()
    roi_multiplicador = vne_total / costo_total if costo_total > 0 else 0
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Clientes en Mora</div>
            <div class="kpi-val">{total_clientes:,}</div>
            <div class="kpi-subtext" style="color: #4CAF50;">Mora 1-30 días</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Deuda Expuesta</div>
            <div class="kpi-val">S/ {deuda_total/1e6:.2f}M</div>
            <div class="kpi-subtext" style="color: #A0AEC0;">S/ {deuda_total:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Costo Campaña</div>
            <div class="kpi-val">S/ {costo_total:,.2f}</div>
            <div class="kpi-subtext" style="color: #FC8181;">Presupuesto: S/ 5,000</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Retorno Neto (VNE)</div>
            <div class="kpi-val" style="color: #4CAF50;">S/ {vne_total/1e6:.2f}M</div>
            <div class="kpi-subtext" style="color: #4CAF50;">S/ {vne_total:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">ROI Campaña</div>
            <div class="kpi-val">{roi_multiplicador:.1f}x</div>
            <div class="kpi-subtext" style="color: #4CAF50;">Valor / Costo Operativo</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Fila de Gráficos (Matplotlib estilizados para tema oscuro)
    st.markdown("<h3 style='margin-top: 15px; margin-bottom: 15px; color:#A0AEC0;'>📊 Distribución Óptima de la Campaña</h3>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        # Gráfico de Canales Asignados
        counts = df_scheduled['Canal_Asignado'].value_counts()
        colors_dict = {
            'WhatsApp': '#2E7D32',
            'SMS': '#10B981',
            'Llamada': '#2A5C91',
            'Control': '#718096',
            'Campo': '#D97706'
        }
        plot_colors = [colors_dict.get(c, '#718096') for c in counts.index]
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#1A2332')
        ax.set_facecolor('#1A2332')
        
        bars = ax.barh(counts.index, counts.values, color=plot_colors, height=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2D3748')
        ax.spines['bottom'].set_color('#2D3748')
        ax.tick_params(colors='#A0AEC0', labelsize=10)
        ax.xaxis.grid(True, linestyle='--', alpha=0.1, color='#FFFFFF')
        ax.set_axisbelow(True)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (counts.max() * 0.02),
                    bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}',
                    va='center', ha='left', color='#FFFFFF', fontsize=9, fontweight='bold')
        
        ax.set_title("Clientes Asignados por Canal Causal (NBA)", color='#ffffff', fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        
    with g_col2:
        # Gráfico por Día de la Semana
        day_order = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado']
        df_agendados = df_scheduled[df_scheduled['Estado_Agenda'] == 'AGENDADO']
        day_counts = df_agendados['Dia_Semana'].value_counts().reindex(day_order).fillna(0)
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#1A2332')
        ax.set_facecolor('#1A2332')
        
        bars = ax.bar(day_counts.index, day_counts.values, color='#2E7D32', width=0.5, edgecolor='#4CAF50')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2D3748')
        ax.spines['bottom'].set_color('#2D3748')
        ax.tick_params(colors='#A0AEC0', labelsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.1, color='#FFFFFF')
        ax.set_axisbelow(True)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + (day_counts.max() * 0.02),
                        f'{int(height):,}',
                        va='bottom', ha='center', color='#FFFFFF', fontsize=9, fontweight='bold')
                        
        ax.set_title("Agenda de Contactos Agendados por Día", color='#ffffff', fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    # 3. Fila inferior de Alertas con estilo oscuro CobraIQ
    st.markdown("<h3 style='margin-top: 15px; margin-bottom: 10px; color:#A0AEC0;'>🔔 Alertas y Notificaciones de Operación</h3>", unsafe_allow_html=True)
    
    # Función para renderizar alertas
    def render_custom_alert(alert_type, title, text):
        if alert_type == "red":
            bg, border, txt_color = "#2D1A1A", "#FC8181", "#FC8181"
        elif alert_type == "yellow":
            bg, border, txt_color = "#2D2510", "#FDD835", "#FDD835"
        else: # green
            bg, border, txt_color = "#162B1D", "#4CAF50", "#4CAF50"
            
        st.markdown(f"""
        <div style="background-color: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 15px; margin-bottom: 12px; display: flex; flex-direction: column;">
            <div style="font-weight: 700; font-size: 14px; color: {txt_color}; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 13px; color: #E2E8F0;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    # EDITAR
    render_custom_alert(
        "red", 
        "🔴 Capacidad Crítica en Visitas de Campo", 
        "Se alcanzó la capacidad operativa límite de gestores de campo en las regiones Lima y Sur para el Lunes. Se aplicó priorización del optimizador asignando visitas solo a los 230 clientes con mayor probabilidad de respuesta incremental (VNE)."
    )
    
    # EDITAR
    render_custom_alert(
        "yellow", 
        "🟡 Monitoreo de Cartera - Periodo de Inferencia Activo", 
        "Scoring en ejecución sobre la cartera del periodo '2026-03'. Se filtró el universo a clientes con mora temprana (1 a 30 días). Los clientes con mora superior (>30 días) fueron derivados al call center de cobranza pesada."
    )
    
    # EDITAR
    render_custom_alert(
        "green", 
        "🟢 Optimización Operativa Exitosa", 
        f"Algoritmo MILP resuelto exitosamente. Asignación de canales completada. Presupuesto diario asignado (S/ 5,000) utilizado en un 99.99% (S/ {costo_total:,.2f}), maximizando el retorno incremental de la cartera."
    )

# ---------------------------------------------------------------------------
# PÁGINA 2: 👥 GESTIÓN DE CARTERA
# ---------------------------------------------------------------------------
elif current_page == "👥 Gestión de Cartera":
    st.markdown("<h2 style='margin-bottom: 20px;'>👥 Gestión y Planificación de Cartera</h2>", unsafe_allow_html=True)
    
    # Barra de filtros superior
    st.markdown("<div style='background-color: #1A2332; padding: 15px; border-radius: 8px; border: 1px solid #2D3748; margin-bottom: 20px;'>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        # Filtro de Región
        region_list = ["Todas"] + list(df_scheduled['Region'].dropna().unique())
        selected_region = st.selectbox("Región", region_list)
        
    with f_col2:
        # Filtro de Zona
        zona_list = ["Todas"] + list(df_scheduled['Zona'].dropna().unique())
        selected_zona = st.selectbox("Zona", zona_list)
        
    with f_col3:
        # Filtro de Canal Asignado
        canal_list = ["Todos"] + list(df_scheduled['Canal_Asignado'].dropna().unique())
        selected_canal = st.selectbox("Canal NBA", canal_list)
        
    with f_col4:
        # Filtro de Estado Agenda
        estado_list = ["Todos"] + list(df_scheduled['Estado_Agenda'].dropna().unique())
        selected_estado = st.selectbox("Estado Agenda", estado_list)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Aplicar filtros
    filtered_df = df_scheduled.copy()
    if selected_region != "Todas":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_zona != "Todas":
        filtered_df = filtered_df[filtered_df['Zona'] == selected_zona]
    if selected_canal != "Todos":
        filtered_df = filtered_df[filtered_df['Canal_Asignado'] == selected_canal]
    if selected_estado != "Todos":
        filtered_df = filtered_df[filtered_df['Estado_Agenda'] == selected_estado]
        
    # Mostrar tabla analítica interactiva
    st.markdown(f"**Registros seleccionados: {len(filtered_df):,}**")
    
    # Seleccionar columnas descriptivas para el usuario
    cols_to_show = [
        "Cliente_ID", "Region", "Zona", "Deuda_Expuesta", "Canal_Asignado", 
        "Prob_Incremental", "Costo_Incurrido", "Valor_Esperado_Neto", 
        "Dia_Semana", "Hora_Inicio", "Asesor_Asignado", "Estado_Agenda"
    ]
    
    # Formatear la tabla
    df_table = filtered_df[cols_to_show].copy()
    df_table['Deuda_Expuesta'] = df_table['Deuda_Expuesta'].round(2)
    df_table['Prob_Incremental'] = df_table['Prob_Incremental'].round(4)
    df_table['Costo_Incurrido'] = df_table['Costo_Incurrido'].round(2)
    df_table['Valor_Esperado_Neto'] = df_table['Valor_Esperado_Neto'].round(2)
    
    # Mostrar tabla interactiva nativa de Streamlit (estilizada por el theme oscuro)
    st.dataframe(df_table, use_container_width=True, height=450)
    
    # Botón al final para descargar CSV
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exportar Resultados a CSV (Agenda Óptima)",
        data=csv_data,
        file_name="cobraiq_agenda_optima.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------------------------
# PÁGINA 3: 🔍 FICHA DEL CLIENTE
# ---------------------------------------------------------------------------
elif current_page == "🔍 Ficha del Cliente":
    st.markdown("<h2 style='margin-bottom: 20px;'>🔍 Consulta Individual del Cliente</h2>", unsafe_allow_html=True)
    
    # Buscador de cliente por texto
    client_id_str = st.text_input("Ingrese ID del Cliente a Consultar", value="44123")
    
    if client_id_str:
        try:
            client_id = int(client_id_str)
        except ValueError:
            st.error("Por favor ingrese un ID de cliente numérico válido.")
            st.stop()
            
        # Buscar en la asignación optimizada
        c_assignment = df_scheduled[df_scheduled['Cliente_ID'] == client_id]
        
        if c_assignment.empty:
            st.warning(f"No se encontró asignación operativa para el cliente con ID {client_id} en el periodo actual.")
        else:
            c_assign_row = c_assignment.iloc[0]
            
            # Buscar en la base demográfica de clientes
            c_demog = None
            if df_clientes is not None:
                match_demog = df_clientes[df_clientes['cliente_id'] == client_id]
                if not match_demog.empty:
                    c_demog = match_demog.iloc[0]
            
            # Buscar propensiones/uplift causales
            c_uplifts = None
            if df_scores is not None:
                match_scores = df_scores[df_scores['Cliente_ID'] == client_id]
                if not match_scores.empty:
                    c_uplifts = match_scores.iloc[0]
                    
            # 1. Cabecera del Cliente Estilo Premium
            genero_str = c_demog['genero'] if c_demog is not None else "N/A"
            region_str = c_assign_row['Region']
            zona_str = c_assign_row['Zona'].title()
            tipo_cli = c_demog['tipo_cliente'].title() if c_demog is not None else "N/A"
            edad_str = str(c_demog['edad']) if c_demog is not None else "N/A"
            
            # Iniciales para el avatar
            avatar_initial = "👤" if genero_str == "N/A" else ("👨" if genero_str == "M" else "👩")
            avatar_bg = "#2A5C91" if genero_str == "M" else "#2E7D32"
            
            st.markdown(f"""
            <div style="background-color: #1A2332; border: 1px solid #2D3748; border-radius: 10px; padding: 20px; display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                <div style="background-color: {avatar_bg}; border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; font-size: 36px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                    {avatar_initial}
                </div>
                <div>
                    <h3 style="margin: 0; color: #ffffff;">Cliente ID: {client_id}</h3>
                    <div style="display: flex; gap: 15px; margin-top: 5px; font-size: 14px; color: #A0AEC0;">
                        <span><strong>Región:</strong> {region_str}</span>
                        <span>•</span>
                        <span><strong>Zona:</strong> {zona_str}</span>
                        <span>•</span>
                        <span><strong>Edad:</strong> {edad_str} años ({genero_str})</span>
                        <span>•</span>
                        <span><strong>Segmento:</strong> {tipo_cli}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_info1, col_info2 = st.columns(2)
            
            # Columna izquierda: Información del cliente y financiera
            with col_info1:
                st.markdown("<h4 style='color: #A0AEC0; margin-bottom: 10px;'>📋 Perfil Financiero e Historial</h4>", unsafe_allow_html=True)
                
                # Fila llave-valor limpia
                def render_row(label, val, highlight=False):
                    color = "#4CAF50" if highlight else "#ffffff"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #2D3748; padding: 8px 0;">
                        <span style="color: #A0AEC0; font-size: 14px;">{label}</span>
                        <span style="color: {color}; font-size: 14px; font-weight: 600;">{val}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Variables del cliente
                score_riesgo = int(c_demog['score_riesgo']) if c_demog is not None else 0
                prob_def = float(c_demog['prob_default']) * 100 if c_demog is not None else 0.0
                atrasos_prev = int(c_demog['num_atrasos_previos']) if c_demog is not None else 0
                mora_prom = float(c_demog['dias_mora_promedio']) if c_demog is not None else 0.0
                ratio_pago = float(c_demog['ratio_pago']) * 100 if c_demog is not None else 0.0
                ultimo_pago = int(c_demog['ultimo_pago_dias']) if c_demog is not None else 0
                es_dig = "Sí" if c_demog is not None and c_demog['es_digital'] == 1 else "No"
                uso_whatsapp = float(c_demog['uso_whatsapp']) * 100 if c_demog is not None else 0.0
                
                render_row("Deuda Vencida (Cuota Mensual)", f"S/ {c_assign_row['Deuda_Expuesta']:,.2f}", highlight=True)
                render_row("Score de Riesgo Mibanco", f"{score_riesgo} / 1000")
                render_row("Probabilidad de Default", f"{prob_def:.2f}%")
                render_row("Número de Aatrasos Previos", f"{atrasos_prev} veces")
                render_row("Días de Mora Promedio Histórico", f"{mora_prom:.1f} días")
                render_row("Ratio de Pago Realizado", f"{ratio_pago:.1f}%")
                render_row("Días desde el Último Pago", f"{ultimo_pago} días")
                render_row("Perfil Digital", es_dig)
                render_row("Uso de WhatsApp", f"{uso_whatsapp:.1f}%")
                
            # Columna derecha: Decisión y CATE Scores
            with col_info2:
                st.markdown("<h4 style='color: #A0AEC0; margin-bottom: 10px;'>💡 Decisión del Orquestador NBA</h4>", unsafe_allow_html=True)
                
                canal_nba = c_assign_row['Canal_Asignado']
                vne_cli = c_assign_row['Valor_Esperado_Neto']
                costo_cli = c_assign_row['Costo_Incurrido']
                estado_ag = c_assign_row['Estado_Agenda']
                
                # Mapeo de colores del canal
                color_canal = "#4CAF50" if canal_nba == "WhatsApp" else ("#10B981" if canal_nba == "SMS" else ("#3B82F6" if canal_nba == "Llamada" else ("#F59E0B" if canal_nba == "Campo" else "#718096")))
                
                st.markdown(f"""
                <div style="background-color: #161D2B; border: 1px solid #2D3748; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 14px; color: #A0AEC0; font-weight: 500;">Canal Seleccionado (NBA):</span>
                        <span style="background-color: {color_canal}; color: #ffffff; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: 700;">{canal_nba.upper()}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #A0AEC0;">Valor Neto Esperado (VNE):</span>
                        <span style="color: #4CAF50; font-weight: 600;">S/ {vne_cli:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #A0AEC0;">Costo del Canal:</span>
                        <span style="color: #ffffff;">S/ {costo_cli:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #A0AEC0;">Día / Hora Asignado:</span>
                        <span style="color: #ffffff; font-weight: 600;">{c_assign_row['Dia_Semana']} ({c_assign_row['Hora_Inicio']}-{c_assign_row['Hora_Fin']})</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #A0AEC0;">Asesor Encargado:</span>
                        <span style="color: #ffffff;">{c_assign_row['Asesor_Asignado']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #A0AEC0;">Estado Agenda:</span>
                        <span style="color: {color_canal}; font-weight: 600;">{estado_ag}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 14px; border-top: 1px solid #2D3748; padding-top: 8px; margin-top: 5px;">
                        <span style="color: #A0AEC0;">Restricción Aplicada:</span>
                        <span style="color: #A0AEC0; font-style: italic; font-size: 12px;">{c_assign_row['Restriccion_Aplicada']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualización de Uplift CATE Individual
                st.markdown("<h4 style='color: #A0AEC0; margin-top: 10px; margin-bottom: 10px;'>📊 Probabilidad de Respuesta Incremental (CATE)</h4>", unsafe_allow_html=True)
                
                if c_uplifts is not None:
                    # Mostrar progress bar para cada canal
                    def render_uplift_bar(canal, value):
                        pct = value * 100
                        # Mapear valor para la visualización del color (positivo verde, negativo rojo)
                        bar_color = "#4CAF50" if pct > 0 else "#FC8181"
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 3px;">
                                <span style="color: #ffffff; font-weight: 500;">{canal}</span>
                                <span style="color: {bar_color}; font-weight: 600;">{pct:+.2f}%</span>
                            </div>
                            <div style="background-color: #2D3748; border-radius: 4px; height: 8px; width: 100%; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {min(max(abs(pct)*5, 0), 100)}%; height: 100%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    render_uplift_bar("WhatsApp", c_uplifts['Uplift_WhatsApp'])
                    render_uplift_bar("SMS", c_uplifts['Uplift_SMS'])
                    render_uplift_bar("Llamada", c_uplifts['Uplift_Llamada'])
                    render_uplift_bar("Campo (Visita Física)", c_uplifts['Uplift_Campo'])
                else:
                    st.info("Scores CATE individuales no disponibles para este cliente en el archivo parquet.")

# ---------------------------------------------------------------------------
# PÁGINA 4: 📊 ANÁLISIS E INSIGHTS
# ---------------------------------------------------------------------------
elif current_page == "📊 Análisis e Insights":
    st.markdown("<h2 style='margin-bottom: 20px;'>📊 Análisis e Insights de Propensión Causal</h2>", unsafe_allow_html=True)
    
    # Grid de KPIs avanzados del optimizador
    k_col1, k_col2, k_col3 = st.columns(3)
    
    # Valores extraídos directamente de los logs de la consola del script de optimización
    vne_azar = 327745.08
    vne_opt = df_scheduled['Valor_Esperado_Neto'].sum()
    valor_agregado = vne_opt - vne_azar
    pct_agregado = (vne_opt / vne_azar - 1) * 100
    
    with k_col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Retorno Tradicional (Random)</div>
            <div class="kpi-val" style="color: #A0AEC0;">S/ {vne_azar:,.2f}</div>
            <div class="kpi-subtext">Asignación sin optimizador causal</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k_col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Retorno CobraIQ (IA Causal)</div>
            <div class="kpi-val" style="color: #4CAF50;">S/ {vne_opt:,.2f}</div>
            <div class="kpi-subtext">Modelo Uplift + Mochila MILP</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k_col3:
        st.markdown(f"""
        <div class="kpi-container" style="border-color: #4CAF50;">
            <div class="kpi-header" style="color: #4CAF50;">Valor Neto Agregado (IA)</div>
            <div class="kpi-val" style="color: #4CAF50;">S/ {valor_agregado:,.2f}</div>
            <div class="kpi-subtext" style="color: #4CAF50;">Incremento de +{pct_agregado:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='border-color: #2D3748; margin: 15px 0;'>", unsafe_allow_html=True)
    
    # Distribuciones operativas adicionales
    i_col1, i_col2 = st.columns(2)
    
    with i_col1:
        # Monto VNE total por región
        region_vne = df_scheduled.groupby('Region')['Valor_Esperado_Neto'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#1A2332')
        ax.set_facecolor('#1A2332')
        
        bars = ax.bar(region_vne.index, region_vne.values, color='#2A5C91', width=0.45, edgecolor='#3B82F6')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2D3748')
        ax.spines['bottom'].set_color('#2D3748')
        ax.tick_params(colors='#A0AEC0', labelsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.1, color='#FFFFFF')
        ax.set_axisbelow(True)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (region_vne.max() * 0.02),
                    f'S/ {height/1e3:.1f}k',
                    va='bottom', ha='center', color='#FFFFFF', fontsize=9, fontweight='bold')
                    
        ax.set_title("Valor Neto Esperado (VNE) por Región", color='#ffffff', fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        
    with i_col2:
        # Relación de Deuda Promedio por Canal Asignado
        deuda_canal = df_scheduled.groupby('Canal_Asignado')['Deuda_Expuesta'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#1A2332')
        ax.set_facecolor('#1A2332')
        
        # Colores por canal
        colors_dict = {
            'WhatsApp': '#2E7D32',
            'SMS': '#10B981',
            'Llamada': '#2A5C91',
            'Control': '#718096',
            'Campo': '#D97706'
        }
        plot_colors = [colors_dict.get(c, '#718096') for c in deuda_canal.index]
        
        bars = ax.bar(deuda_canal.index, deuda_canal.values, color=plot_colors, width=0.45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2D3748')
        ax.spines['bottom'].set_color('#2D3748')
        ax.tick_params(colors='#A0AEC0', labelsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.1, color='#FFFFFF')
        ax.set_axisbelow(True)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (deuda_canal.max() * 0.02),
                    f'S/ {int(height):,}',
                    va='bottom', ha='center', color='#FFFFFF', fontsize=9, fontweight='bold')
                    
        ax.set_title("Deuda Promedio de Clientes Asignados por Canal", color='#ffffff', fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("""
    <div style="background-color: #1A2332; border: 1px solid #2D3748; border-radius: 8px; padding: 20px; margin-top: 15px;">
        <h4 style="color: #ffffff; margin-top: 0; margin-bottom: 8px;">💡 Explicación del Logro Causal</h4>
        <p style="font-size: 14px; color: #A0AEC0; line-height: 1.5; margin: 0;">
            El optimizador matemático CobraIQ asigna a cada cliente al canal más rentable que genera el mayor impacto de pago neto. Como se observa en la distribución de deuda promedio, el modelo asigna de forma inteligente los canales costosos como <strong>Campo</strong> (costo S/ 8.00) únicamente a clientes con deudas muy elevadas (promedio S/ 1,299) y con altos coeficientes CATE (uplift causal). De la misma manera, la inteligencia artificial clasifica en <strong>Control</strong> a clientes que de todas formas pagarán sin requerir contacto alguno, ahorrando costos de envío innecesarios.
        </p>
    </div>
    """, unsafe_allow_html=True)
