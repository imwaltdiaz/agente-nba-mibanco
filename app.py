"""
app.py - Frontend CobraIQ (Fase D)
========================================================================
Aplicación web interactiva que funciona como interfaz del orquestador NBA.
Clon en diseño, colores y pestañas de la identidad institucional de MiBanco.
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
import plotly.express as px
import plotly.graph_objects as go

# Agregar raíz del proyecto al path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
import src.config as config

# Configuración de página
st.set_page_config(
    page_title="CobraIQ - Control de Cobranza Inteligente",
    page_icon="💚",
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


# (get_saturation_data fue eliminado por limpieza de gráficos antiguos)


# Carga de datasets
df_scheduled, file_name_loaded = get_latest_data()
df_clientes = load_client_database()
df_scores = load_cate_scores()

# ---------------------------------------------------------------------------
# 2. DISEÑO Y ESTILIZACIÓN (CSS PERSONALIZADO - IDENTIDAD CLARA MIBANCO)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Inyección de tipografía moderna */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FAFAFA !important;
        color: #1A3A2A !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Configuración del Sidebar lateral */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1.5px solid #D5E8DC !important;
    }
    
    /* Habilitar scroll del sidebar y asegurar estilo del texto */
    [data-testid="stSidebar"] * {
        color: #1A3A2A !important;
    }
    
    /* Ajustes generales de inputs y sliders */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1A3A2A !important;
        border: 1.5px solid #D5E8DC !important;
        border-radius: 8px !important;
    }
    
    input {
        background-color: #ffffff !important;
        color: #1A3A2A !important;
        border: 1.5px solid #D5E8DC !important;
        border-radius: 8px !important;
    }
    
    /* Menú de Navegación Vertical con Radio Buttons */
    div[role="radiogroup"] {
        background-color: transparent !important;
        gap: 8px !important;
        padding-top: 15px;
    }
    div[role="radiogroup"] label {
        background-color: #ffffff !important;
        border: 1.5px solid #D5E8DC !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        color: #7A9088 !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        display: flex;
        align-items: center;
        width: 100% !important;
    }
    
    /* Ocultar el circulo de seleccion nativo */
    div[role="radiogroup"] label div[class*="StyledRadio"] {
        display: none !important;
    }
    
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #1B8C3E !important;
        border-color: #1B8C3E !important;
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(27, 140, 98, 0.2);
    }
    
    div[role="radiogroup"] label[data-checked="true"] span {
        color: #FFFFFF !important;
    }
    
    div[role="radiogroup"] label:hover {
        background-color: #F0F7F3 !important;
        border-color: #1B8C3E !important;
        color: #1B8C3E !important;
    }
    
    div[role="radiogroup"] label span {
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Tarjetas KPI y contenedores rounded estilo MiBanco */
    .kpi-container {
        background-color: #ffffff !important;
        border: 1.5px solid #D5E8DC !important;
        border-radius: 12px !important;
        padding: 20px !important;
        text-align: center !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 15px !important;
    }
    
    .kpi-header {
        color: #7A9088 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        margin-bottom: 6px !important;
        letter-spacing: 0.8px !important;
    }
    
    .kpi-val {
        color: #1A3A2A !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    
    .kpi-subtext {
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Contenedor principal de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border: 1.5px solid #D5E8DC !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7A9088 !important;
        background-color: transparent !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background-color: #1B8C3E !important;
    }
    
    /* Cabecera institucional con gradiente */
    .mibanco-header {
        background: linear-gradient(135deg, #1B8C3E 0%, #25A84E 40%, #4FC97A 70%, #7EDBA0 100%) !important;
        padding: 2rem !important;
        border-radius: 16px !important;
        color: white !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 4px 15px rgba(27, 140, 62, 0.15) !important;
    }
    .mibanco-header h1 {
        color: white !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .mibanco-header p {
        color: white !important;
        opacity: 0.95 !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
    }
    
    /* Estilo para el botón de Cerrar Sesión en el Sidebar */
    .sidebar-logout-container div.stButton > button {
        background-color: transparent !important;
        color: #7A9088 !important;
        border: 1.5px solid #D5E8DC !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        width: 100% !important;
        padding: 10px 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: none !important;
    }
    
    .sidebar-logout-container div.stButton > button:hover {
        background-color: #FFF5F5 !important;
        color: #C53030 !important;
        border-color: #FEB2B2 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. SIDEBAR DE NAVEGACIÓN Y CONFIGURACIÓN
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logotipo oficial SVG de MiBanco recreado
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; padding: 10px 0 5px 0;">
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">
            <!-- Sol central brillante -->
            <circle cx="18" cy="18" r="5" fill="#FDD700" />
            <!-- Rayos/Curvas del sol institucionales de MiBanco -->
            <path d="M18 4V8" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M18 28V32" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M4 18H8" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M28 18H32" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M8.1 8.1L10.9 10.9" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M25.1 25.1L27.9 27.9" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M8.1 27.9L10.9 25.1" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
            <path d="M25.1 10.9L27.9 8.1" stroke="#FDD700" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <span style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 26px; font-weight: 800; color: #1B8C3E; letter-spacing: -1.2px; margin: 0; display: inline-block; line-height: 1;">
            mibanco
        </span>
    </div>
    <div style="font-size: 10px; color: #7A9088; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-top: -2px; margin-left: 2px;">
        Control de Cobranza Inteligente
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 15px 0; border-color: #D5E8DC;'>", unsafe_allow_html=True)
    
    # Perfil de usuario autenticado
    username = st.session_state.get('username', 'Colaborador')
    user_area = st.session_state.get('user_area', 'Gestión de Cobranzas')
    initials = "".join([part[0] for part in username.split()[:2]]).upper()
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; background-color: #FAFAFA; border: 1px solid #D5E8DC; border-radius: 10px; padding: 10px 12px; margin-top: 5px; margin-bottom: 15px;">
        <div style="background: linear-gradient(135deg, #1B8C3E, #4FC97A); color: #ffffff; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; border: 1px solid #D5E8DC;">
            {initials}
        </div>
        <div style="line-height: 1.2;">
            <div style="font-size: 13px; font-weight: 700; color: #1A3A2A;">{username.split()[0]}</div>
            <div style="font-size: 10.5px; color: #7A9088;">{user_area}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Menú de selección vertical
    current_page = st.radio(
        label="Navegación",
        options=["🏠 Dashboard Ejecutivo", "👥 Gestión de Cartera", "🔍 Ficha del Cliente", "📊 Análisis e Insights"],
        label_visibility="collapsed"
    )
    
    # Meta información en el footer del Sidebar
    st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
    
    if df_scheduled is not None:
        total_inf = len(df_scheduled)
        pct_digital_inf = (df_scheduled['Canal_Asignado'].isin(['WhatsApp', 'SMS']).sum() / total_inf) * 100
        
        # El ahorro estimado es el VNE de la IA menos el VNE del Azar
        vne_opt = df_scheduled['Valor_Esperado_Neto'].sum()
        vne_azar = 327745.08
        ahorro_estimado = vne_opt - vne_azar
        
        st.markdown(f"""
        <div style="background-color: #ffffff; border: 1.5px solid #D5E8DC; border-radius: 12px; padding: 15px; margin-top: 20px;">
            <div style="font-size: 11px; color: #7A9088; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px; border-bottom: 1px solid #D5E8DC; padding-bottom: 5px;">
                Resumen Ejecutivo
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13px; color: #7A9088;">Total Clientes:</span>
                <span style="font-size: 13px; color: #1A3A2A; font-weight: 600;">{total_inf:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13px; color: #7A9088;">Canal Digital:</span>
                <span style="font-size: 13px; color: #1B8C3E; font-weight: 600;">{pct_digital_inf:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 13px; color: #7A9088;">Ahorro Neto IA:</span>
                <span style="font-size: 13px; color: #1B8C3E; font-weight: 600;">S/ {ahorro_estimado / 1e6:.2f}M</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #FFF5F5; border: 1.5px solid #FC8181; border-radius: 12px; padding: 15px; margin-top: 20px;">
            <span style="color: #C53030; font-size: 13px;">Error: Datos no cargados.</span>
        </div>
        """, unsafe_allow_html=True)
        


# ---------------------------------------------------------------------------
# DECLARACIÓN DEL DIALOG MODAL (POP-UP MATEMÁTICO)
# ---------------------------------------------------------------------------

@st.dialog("🔍 Detalle de Cálculos Matemáticos")
def show_math_details(client_id, debt, canal, cost, vne):
    st.markdown(f"### 📋 Métricas del Cliente: `{client_id}`")
    st.write(f"- **Deuda Expuesta**: `S/ {debt:,.2f}` (suma de cuotas vencidas)")
    st.write(f"- **Canal Asignado por IA**: `{canal}` (costo unitario: `S/ {cost:,.2f}`)")
    st.write(f"- **Retorno Neto Causal Esperado (VNE)**: `S/ {vne:,.2f}`")
    
    st.markdown("---")
    st.markdown("### 🧮 Ecuación del Valor Neto Esperado (VNE)")
    st.latex(r"VNE = (\tau_{canal} \times \text{Deuda}) - \text{Costo}_{canal}")
    st.write("Donde:")
    st.markdown("- **$\\tau_{canal}$** es el **Uplift Causal (CATE)** del canal sobre el grupo control (T=0).")
    st.markdown("- **$\\text{Deuda}$** es el monto expuesto al cobro (cuota del cliente).")
    st.markdown("- **$\\text{Costo}_{canal}$** es el costo unitario de ejecución de la cobranza.")
    
    # Cargar detalles CATE
    if df_scores is not None:
        c_scores = df_scores[df_scores['Cliente_ID'] == client_id]
        if not c_scores.empty:
            row = c_scores.iloc[0]
            st.markdown("---")
            st.markdown("### 📊 Desglose de Uplift Causal del Cliente")
            st.write(f"- **WhatsApp**: CATE = `{row['Uplift_WhatsApp']*100:+.2f}%` | Retorno Bruto: `S/ {(row['Uplift_WhatsApp']*debt):+.2f}`")
            st.write(f"- **SMS**: CATE = `{row['Uplift_SMS']*100:+.2f}%` | Retorno Bruto: `S/ {(row['Uplift_SMS']*debt):+.2f}`")
            st.write(f"- **Llamada**: CATE = `{row['Uplift_Llamada']*100:+.2f}%` | Retorno Bruto: `S/ {(row['Uplift_Llamada']*debt):+.2f}`")
            st.write(f"- **Campo (Visita)**: CATE = `{row['Uplift_Campo']*100:+.2f}%` | Retorno Bruto: `S/ {(row['Uplift_Campo']*debt):+.2f}`")
            
            # Cálculo detallado paso a paso para el canal óptimo
            canal_col_map = {
                'WhatsApp': 'Uplift_WhatsApp',
                'SMS': 'Uplift_SMS',
                'Llamada': 'Uplift_Llamada',
                'Campo': 'Uplift_Campo'
            }
            col_name = canal_col_map.get(canal)
            if col_name and col_name in row:
                tau_val = row[col_name]
                retorno_bruto = tau_val * debt
                calculo_vne = retorno_bruto - cost
                
                st.markdown("---")
                st.markdown(f"### ⚙️ Cálculo Detallado del Canal Óptimo ({canal})")
                st.latex(rf"\text{{VNE}} = ({tau_val:+.4f} \times S/ {debt:,.2f}) - S/ {cost:,.2f}")
                st.latex(rf"\text{{VNE}} = S/ {retorno_bruto:,.2f} - S/ {cost:,.2f} = S/ {calculo_vne:,.2f}")
                st.markdown(f"""
                - **Uplift Causal ($\tau$):** `{tau_val*100:+.2f}%`
                - **Ingreso Causal Incremental Esperado (Retorno Bruto):** `S/ {retorno_bruto:,.2f}`
                - **Costo Operativo del Canal:** `S/ {cost:,.2f}`
                - **Valor Neto Esperado (VNE) Final:** `S/ {calculo_vne:,.2f}`
                """)

# ---------------------------------------------------------------------------
# 5. LÓGICA Y CONTENIDO POR PANTALLA
# ---------------------------------------------------------------------------

if df_scheduled is None:
    st.error("No se pudieron cargar los datos de asignación. Asegúrate de ejecutar primero optimizer.py.")
    st.stop()

# ---------------------------------------------------------------------------
# PÁGINA 1: 🏠 DASHBOARD EJECUTIVO
# ---------------------------------------------------------------------------
if current_page == "🏠 Dashboard Ejecutivo":
    # Cabecera institucional con gradiente
    st.markdown("""
    <div class="mibanco-header">
        <h1>🟢 Motor de Decisiones NBA - Mibanco</h1>
        <p>Cobranza Inteligente: Optimización Causal del Canal y Momento de Contacto de Cartera</p>
    </div>
    """, unsafe_allow_html=True)
    
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
            <div class="kpi-subtext" style="color: #1B8C3E;">Mora 1-30 días</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Deuda Expuesta</div>
            <div class="kpi-val">S/ {deuda_total/1e6:.2f}M</div>
            <div class="kpi-subtext" style="color: #7A9088;">S/ {deuda_total:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Costo Campaña</div>
            <div class="kpi-val">S/ {costo_total:,.2f}</div>
            <div class="kpi-subtext" style="color: #C53030;">Presupuesto: S/ 5,000</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Retorno Neto (VNE)</div>
            <div class="kpi-val" style="color: #1B8C3E;">S/ {vne_total/1e6:.2f}M</div>
            <div class="kpi-subtext" style="color: #1B8C3E;">S/ {vne_total:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">ROI Campaña</div>
            <div class="kpi-val">{roi_multiplicador:.1f}x</div>
            <div class="kpi-subtext" style="color: #1B8C3E;">Valor / Costo Operativo</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Fila de Gráficos (Heatmap de Plotly + Horizontal Bar Chart Plotly)
    st.markdown("<h3 style='margin-top: 15px; margin-bottom: 15px; color:#1A3A2A;'>📊 Distribución Óptima de la Campaña</h3>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        # Gráfico A: Clientes Asignados por Canal Causal con Plotly (Homogeneizado)
        counts = df_scheduled['Canal_Asignado'].value_counts().reset_index()
        counts.columns = ['Canal', 'Clientes']
        
        colors_map = {
            'WhatsApp': '#1B8C3E',
            'SMS': '#4FC97A',
            'Llamada': '#2A5C91',
            'Control': '#7A9088',
            'Campo': '#D97706'
        }
        
        fig_bar = px.bar(
            counts,
            x='Clientes',
            y='Canal',
            orientation='h',
            color='Canal',
            color_discrete_map=colors_map,
            text_auto=True
        )
        
        fig_bar.update_layout(
            title=dict(
                text="Clientes Asignados por Canal Causal (NBA)",
                font=dict(size=14, color="#1A3A2A", family="Outfit", weight="bold")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit, sans-serif", size=12, color="#1A3A2A"),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E5F3EB", showline=True, linecolor="#D5E8DC", tickcolor="#D5E8DC"),
            yaxis=dict(showgrid=False, showline=True, linecolor="#D5E8DC", tickcolor="#D5E8DC", categoryorder="total ascending"),
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with g_col2:
        # Gráfico B: Heatmap de Agenda usando Plotly
        df_ag = df_scheduled[df_scheduled['Estado_Agenda'] == 'AGENDADO']
        day_order = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado']
        
        # Agrupar por Canal y Día
        df_counts = df_ag.groupby(['Canal_Asignado', 'Dia_Semana']).size().reset_index(name='Contactos')
        
        # Pipotear
        pivot_df = df_counts.pivot(index='Canal_Asignado', columns='Dia_Semana', values='Contactos').fillna(0)
        
        # Reordenar ejes
        available_days = [d for d in day_order if d in pivot_df.columns]
        pivot_df = pivot_df[available_days]
        
        channels_order = ['WhatsApp', 'SMS', 'Llamada', 'Campo']
        available_channels = [c for c in channels_order if c in pivot_df.index]
        pivot_df = pivot_df.reindex(available_channels, fill_value=0)
        
        # Crear Heatmap de Plotly
        fig_heat = px.imshow(
            pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale="Greens",
            aspect="auto",
            labels=dict(x="Día de la Semana", y="Canal de Contacto", color="Volumen"),
            text_auto=True
        )
        
        fig_heat.update_layout(
            title=dict(
                text="Mapa de Calor de Agenda de Contactabilidad",
                font=dict(size=14, color="#1A3A2A", family="Outfit", weight="bold")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit, sans-serif", size=12, color="#1A3A2A"),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=False, tickcolor="#D5E8DC"),
            yaxis=dict(showgrid=False, tickcolor="#D5E8DC")
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# PÁGINA 2: 👥 GESTIÓN DE CARTERA
# ---------------------------------------------------------------------------
elif current_page == "👥 Gestión de Cartera":
    st.markdown("<h2 style='margin-bottom: 20px; color: #1A3A2A;'>👥 Gestión y Planificación de Cartera</h2>", unsafe_allow_html=True)
    
    # Barra de filtros superior
    st.markdown("<div style='background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1.5px solid #D5E8DC; margin-bottom: 20px;'>", unsafe_allow_html=True)
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
    
    # Mostrar tabla interactiva nativa de Streamlit (estilizada por el theme claro)
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
    st.markdown("<h2 style='margin-bottom: 20px; color: #1A3A2A;'>🔍 Consulta Individual del Cliente</h2>", unsafe_allow_html=True)
    
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
            avatar_bg = "#E5F3EB"
            
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1.5px solid #D5E8DC; border-radius: 12px; padding: 20px; display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
                <div style="background-color: {avatar_bg}; border-radius: 50%; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; font-size: 36px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1.5px solid #D5E8DC;">
                    {avatar_initial}
                </div>
                <div>
                    <h3 style="margin: 0; color: #1A3A2A;">Cliente ID: {client_id}</h3>
                    <div style="display: flex; gap: 15px; margin-top: 5px; font-size: 14px; color: #7A9088;">
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
                st.markdown("<h4 style='color: #1A3A2A; margin-bottom: 10px;'>📋 Perfil Financiero e Historial</h4>", unsafe_allow_html=True)
                
                # Fila llave-valor limpia
                def render_row(label, val, highlight=False):
                    color = "#1B8C3E" if highlight else "#1A3A2A"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #D5E8DC; padding: 8px 0;">
                        <span style="color: #7A9088; font-size: 14px;">{label}</span>
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
                render_row("Número de Atrasos Previos", f"{atrasos_prev} veces")
                render_row("Días de Mora Promedio Histórico", f"{mora_prom:.1f} días")
                render_row("Ratio de Pago Realizado", f"{ratio_pago:.1f}%")
                render_row("Días desde el Último Pago", f"{ultimo_pago} días")
                render_row("Perfil Digital", es_dig)
                render_row("Uso de WhatsApp", f"{uso_whatsapp:.1f}%")
                
            # Columna derecha: Decisión y CATE Scores
            with col_info2:
                st.markdown("<h4 style='color: #1A3A2A; margin-bottom: 10px;'>💡 Decisión del Orquestador NBA</h4>", unsafe_allow_html=True)
                
                canal_nba = c_assign_row['Canal_Asignado']
                vne_cli = c_assign_row['Valor_Esperado_Neto']
                costo_cli = c_assign_row['Costo_Incurrido']
                estado_ag = c_assign_row['Estado_Agenda']
                
                # Mapeo de colores del canal
                color_canal = "#1B8C3E" if canal_nba == "WhatsApp" else ("#4FC97A" if canal_nba == "SMS" else ("#2A5C91" if canal_nba == "Llamada" else ("#D97706" if canal_nba == "Campo" else "#7A9088")))
                
                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1.5px solid #D5E8DC; border-radius: 12px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 14px; color: #7A9088; font-weight: 500;">Canal Seleccionado (NBA):</span>
                        <span style="background-color: {color_canal}; color: #ffffff; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: 700;">{canal_nba.upper()}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #7A9088;">Valor Neto Esperado (VNE):</span>
                        <span style="color: #1B8C3E; font-weight: 600;">S/ {vne_cli:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #7A9088;">Costo del Canal:</span>
                        <span style="color: #1A3A2A;">S/ {costo_cli:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #7A9088;">Día / Hora Asignado:</span>
                        <span style="color: #1A3A2A; font-weight: 600;">{c_assign_row['Dia_Semana']} ({c_assign_row['Hora_Inicio']}-{c_assign_row['Hora_Fin']})</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #7A9088;">Asesor Encargado:</span>
                        <span style="color: #1A3A2A;">{c_assign_row['Asesor_Asignado']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px;">
                        <span style="color: #7A9088;">Estado Agenda:</span>
                        <span style="color: {color_canal}; font-weight: 600;">{estado_ag}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 14px; border-top: 1px solid #D5E8DC; padding-top: 8px; margin-top: 5px;">
                        <span style="color: #7A9088;">Restricción Aplicada:</span>
                        <span style="color: #7A9088; font-style: italic; font-size: 12px;">{c_assign_row['Restriccion_Aplicada']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Botón de ver desglose matemático en Pop-up
                if st.button("🔍 Ver Detalle de Cálculos Matemáticos", use_container_width=True):
                    show_math_details(client_id, c_assign_row['Deuda_Expuesta'], canal_nba, costo_cli, vne_cli)
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                # Visualización de Uplift CATE Individual
                st.markdown("<h4 style='color: #1A3A2A; margin-top: 10px; margin-bottom: 10px;'>📊 Probabilidad de Respuesta Incremental (CATE)</h4>", unsafe_allow_html=True)
                
                if c_uplifts is not None:
                    # Mostrar progress bar proporcional para cada canal
                    def render_uplift_bar(canal, value):
                        pct = value * 100
                        bar_color = "#1B8C3E" if pct > 0 else "#FC8181"
                        # Modificación CATE proporcional estricta
                        width_pct = min(max(pct, 0.0), 100.0)
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 3px;">
                                <span style="color: #1A3A2A; font-weight: 500;">{canal}</span>
                                <span style="color: {bar_color}; font-weight: 600;">{pct:+.2f}%</span>
                            </div>
                            <div style="background-color: #E5F3EB; border-radius: 4px; height: 8px; width: 100%; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {width_pct}%; height: 100%;"></div>
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
    st.markdown("<h2 style='margin-bottom: 20px; color: #1A3A2A;'>📊 Análisis e Insights de Propensión Causal</h2>", unsafe_allow_html=True)
    
    # Grid de KPIs avanzados del optimizador
    k_col1, k_col2, k_col3 = st.columns(3)
    
    vne_azar = 327745.08
    vne_opt = df_scheduled['Valor_Esperado_Neto'].sum()
    valor_agregado = vne_opt - vne_azar
    pct_agregado = (vne_opt / vne_azar - 1) * 100
    
    with k_col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Retorno Tradicional (Random)</div>
            <div class="kpi-val" style="color: #7A9088;">S/ {vne_azar:,.2f}</div>
            <div class="kpi-subtext">Asignación sin optimizador causal</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k_col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Retorno CobraIQ (IA Causal)</div>
            <div class="kpi-val" style="color: #1B8C3E;">S/ {vne_opt:,.2f}</div>
            <div class="kpi-subtext">Modelo Uplift + Mochila MILP</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k_col3:
        st.markdown(f"""
        <div class="kpi-container" style="border-color: #1B8C3E !important;">
            <div class="kpi-header" style="color: #1B8C3E;">Valor Neto Agregado (IA)</div>
            <div class="kpi-val" style="color: #1B8C3E;">S/ {valor_agregado:,.2f}</div>
            <div class="kpi-subtext" style="color: #1B8C3E;">Incremento de +{pct_agregado:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='border-color: #D5E8DC; margin: 15px 0;'>", unsafe_allow_html=True)
    
    # PESTAÑA INSIGHTS - GRÁFICOS
    st.markdown("### 📈 Curva de Optimización de Costos")
    
    # Calcular y graficar la Curva de Costo vs. Efectividad Acumulada
    def get_cost_effectiveness_data(df):
        # Ordenar por VNE descendente
        df_sorted = df.sort_values(by='Valor_Esperado_Neto', ascending=False).reset_index(drop=True)
        df_sorted['Costo_Acumulado'] = df_sorted['Costo_Incurrido'].cumsum()
        
        # Efecto de pago incremental
        df_sorted['Efecto_Pago_Incremental'] = df_sorted['Valor_Esperado_Neto'] + df_sorted['Costo_Incurrido']
        df_sorted['Efectividad_Acumulada'] = df_sorted['Efecto_Pago_Incremental'].cumsum()
        
        # Normalizar a porcentajes
        total_cost = df_sorted['Costo_Incurrido'].sum()
        total_efectividad = df_sorted['Efecto_Pago_Incremental'].sum()
        
        df_sorted['Costo_Acumulado_Pct'] = (df_sorted['Costo_Acumulado'] / total_cost) * 100
        df_sorted['Efectividad_Acumulada_Pct'] = (df_sorted['Efectividad_Acumulada'] / total_efectividad) * 100
        
        return df_sorted

    df_curve = get_cost_effectiveness_data(df_scheduled)
    
    # Punto óptimo aproximado (90% de costo)
    idx_90 = (df_curve['Costo_Acumulado_Pct'] - 90).abs().idxmin()
    cost_at_90 = df_curve.loc[idx_90, 'Costo_Acumulado_Pct']
    efect_at_90 = df_curve.loc[idx_90, 'Efectividad_Acumulada_Pct']
    
    # Construir gráfico interactivo Plotly de la curva Costo vs Efectividad
    fig_curve = go.Figure()
    
    # Curva CobraIQ
    fig_curve.add_trace(go.Scatter(
        x=df_curve['Costo_Acumulado_Pct'],
        y=df_curve['Efectividad_Acumulada_Pct'],
        mode='lines',
        name='Optimización Causal (CobraIQ)',
        line=dict(color='#1B8C3E', width=3.5),
        hovertemplate='Costo Acumulado: %{x:.1f}%<br>Efectividad: %{y:.1f}%<extra></extra>'
    ))
    
    # Línea diagonal baseline
    fig_curve.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode='lines',
        name='Estrategia Aleatoria (Baseline)',
        line=dict(color='#7A9088', width=2, dash='dash'),
        hovertemplate='Asignación Aleatoria<extra></extra>'
    ))
    
    # Punto de corte óptimo
    fig_curve.add_trace(go.Scatter(
        x=[cost_at_90],
        y=[efect_at_90],
        mode='markers+text',
        name='Corte Óptimo (-10% Costo)',
        text=[f"Corte Óptimo<br>Costo: {cost_at_90:.1f}%<br>Efecto: {efect_at_90:.1f}%"],
        textposition="bottom right",
        marker=dict(size=12, color='#D97706', symbol='star'),
        hovertemplate='Corte Óptimo: Costo %{x:.1f}%, Efectos %{y:.1f}%<extra></extra>'
    ))
    
    fig_curve.update_layout(
        title=dict(
            text="Curva de Costo vs. Efectividad Acumulada de la Cartera",
            font=dict(size=14, color="#1A3A2A", family="Outfit", weight="bold")
        ),
        xaxis=dict(
            title="Costo Operativo Acumulado (%)",
            gridcolor="#E5F3EB",
            showline=True,
            linecolor="#D5E8DC",
            tickcolor="#D5E8DC",
            range=[0, 100]
        ),
        yaxis=dict(
            title="Efectividad Acumulada (%)",
            gridcolor="#E5F3EB",
            showline=True,
            linecolor="#D5E8DC",
            tickcolor="#D5E8DC",
            range=[0, 105]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", size=12, color="#1A3A2A"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_curve, use_container_width=True)

    # 4. NUEVA IMPLEMENTACIÓN: "Gráfico de Retorno/Ahorro por Canal"
    with st.container():
        st.subheader("📊 Análisis de Retorno y Ahorro por Canal de Contacto")
        
        # Cargar los datasets requeridos para el cálculo
        df_cates_proc = df_scores if df_scores is not None else load_cate_scores()
        
        # Cargar el archivo de asignación optimizada específico solicitado
        processed_dir = ROOT_DIR / "data" / "03_processed"
        specific_csv_path = processed_dir / "asignacion_optimizada_v2_agenda_20260623_1821.csv"
        
        if specific_csv_path.exists() and df_cates_proc is not None:
            df_opt_proc = pd.read_csv(specific_csv_path)
            
            # Unir datasets por Cliente_ID
            m_df = pd.merge(df_opt_proc, df_cates_proc, on='Cliente_ID', suffixes=('', '_cate'))
            
            # Filtrar Control, ya que no tiene costo ni contacto activo
            m_df = m_df[m_df['Canal_Asignado'] != 'Control']
            
            # Agrupar métricas por canal
            summary_channel = m_df.groupby('Canal_Asignado').agg(
                Retorno_Neto_VNE=('Valor_Esperado_Neto', 'sum'),
            ).reset_index()
            
            # Traducir los nombres a español si es necesario
            summary_channel['Canal'] = summary_channel['Canal_Asignado']
            
            # Gráfico de barras horizontales utilizando Plotly Express para el Retorno Neto
            fig_bar_retorno = px.bar(
                summary_channel,
                x='Retorno_Neto_VNE',
                y='Canal',
                orientation='h',
                color_discrete_sequence=['#1B8C3E'],  # Verde corporativo de MiBanco
                text_auto='.2s',
                labels={
                    'Retorno_Neto_VNE': 'Retorno Neto (VNE) (S/)',
                    'Canal': 'Canal de Contacto'
                }
            )
            
            fig_bar_retorno.update_layout(
                title=dict(
                    text="Retorno Neto (VNE) Optimizado por Canal de Contacto",
                    font=dict(size=14, color="#1A3A2A", family="Outfit", weight="bold")
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Outfit, sans-serif", size=12, color="#1A3A2A"),
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(showgrid=True, gridcolor="#E5F3EB", showline=True, linecolor="#D5E8DC", tickcolor="#D5E8DC"),
                yaxis=dict(showgrid=False, showline=True, linecolor="#D5E8DC", tickcolor="#D5E8DC", categoryorder="total ascending"),
                showlegend=False
            )
            
            # Mostrar gráfico responsivo
            st.plotly_chart(fig_bar_retorno, use_container_width=True)
        else:
            st.warning("No se pudo cargar el archivo 'cate_scores.parquet' o 'asignacion_optimizada_v2_agenda_20260623_1821.csv' para el cálculo del retorno.")
