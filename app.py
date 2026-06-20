"""
Aplicación de Streamlit para el Demo del Jurado
Presenta los resultados del motor causal y de la optimización en un dashboard interactivo
con restricciones ajustables en tiempo real (Presupuesto, Capacidades).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Configuración de página
st.set_page_config(
    page_title="Mibanco - Next Best Action (NBA) Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos Premium (CSS personalizado en línea con las guías de diseño)
st.markdown("""
<style>
    /* Estilos globales y fuentes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #f7f9fc;
    }
    
    /* Degradados en cabecera principal */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 3rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-weight: 300;
        opacity: 0.85;
        font-size: 1.15rem;
    }
    
    /* Tarjetas de métricas premium */
    .metric-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 14px;
        border-left: 6px solid #203a43;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-card h3 {
        font-size: 1rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .metric-card small {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado Principal
st.markdown("""
<div class="main-header">
    <h1>🏦 Motor de Decisiones NBA - Mibanco</h1>
    <p>Cobranza Inteligente: Optimización Causal del Canal y Momento de Contacto del Cliente</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para restricciones de negocio
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_K-v0z-00zH46d_B1xU8_3u78Wz3t3b30ew&s", width=200) # Imagen placeholder de Mibanco
st.sidebar.markdown("---")
st.sidebar.header("Restricciones de la Campaña")
budget = st.sidebar.slider("Presupuesto Disponible (S/)", min_value=1000, max_value=50000, value=15000, step=1000)
cap_llamadas = st.sidebar.slider("Capacidad de Llamadas (Mensual)", min_value=500, max_value=10000, value=3000, step=500)
cap_campo = st.sidebar.slider("Capacidad de Gestores en Campo", min_value=50, max_value=2000, value=400, step=50)

st.sidebar.markdown("---")
# Botón para correr la optimización
run_opt = st.sidebar.button("🚀 Re-optimizar con PuLP", use_container_width=True)

# Contenedor de Métricas de Negocio
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Recupero Total Esperado</h3>
        <h2 style="color: #10b981;">S/ --</h2>
        <small>Optimizado maximizando el VNE</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Costo Total de Contacto</h3>
        <h2 style="color: #3b82f6;">S/ --</h2>
        <small>Presupuesto utilizado en campaña</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>ROI Estimado de Cobranza</h3>
        <h2 style="color: #f59e0b;">-- %</h2>
        <small>Retorno sobre la inversión operativa</small>
    </div>
    """, unsafe_allow_html=True)

# Secciones del Dashboard
tab1, tab2, tab3 = st.tabs(["📊 Distribución de Campaña", "🔍 Buscador de Clientes", "📂 Resultados y Descargas"])

with tab1:
    st.subheader("Análisis de Asignaciones y Contactos")
    st.info("Esta sección mostrará gráficos de pastel y barras con la distribución óptima de canales y franjas horarias una vez que el optimizador sea programado.")
    # TODO: Mostrar gráficos con matplotlib / seaborn / plotly por los DS

with tab2:
    st.subheader("Perfil de Cliente y Predicciones de Uplift Causal")
    client_id = st.text_input("Ingrese ID del Cliente", "")
    if client_id:
        st.write(f"Cargando perfil del cliente: **{client_id}**")
        # TODO: Implementar consulta individual a la ABT de Inferencia.
        # Mostrar scores de probabilidad de pago para cada canal y momento sugerido por el modelo de uplift.
        st.warning("Cálculos pendientes del modelo causal...")
    else:
        st.info("Ingrese un ID de cliente para visualizar su recomendación personalizada de contacto.")

with tab3:
    st.subheader("Exportación de la Asignación Óptima")
    # TODO: Mostrar la tabla final resultante data/03_processed/assignment_matrix.csv
    st.dataframe(pd.DataFrame(columns=["Cliente ID", "Crédito ID", "Canal NBA Recomendado", "Momento NBA Recomendado", "VNE Estimado"]))
    st.download_button(
        label="Descargar Matriz de Asignación Completa (CSV)", 
        data="Cliente ID,Credito ID,Canal NBA,Momento NBA,VNE", 
        file_name="nba_asignacion_campana.csv", 
        mime="text/csv"
    )
