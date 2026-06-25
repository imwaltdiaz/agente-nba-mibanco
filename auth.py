import json
import random
import os
from pathlib import Path
import streamlit as st

# Ruta al archivo de persistencia de usuarios
USERS_FILE = Path(__file__).resolve().parent / "data" / "users.json"

def load_users():
    """Carga la base de datos de usuarios de CobraIQ/MiBanco."""
    if not USERS_FILE.exists():
        # Crear usuario por defecto
        default_users = [{
            "email": "admin@mibanco.com.pe",
            "password": "Mibanco2026",
            "nombre": "Administrador CobraIQ",
            "area": "Gestión de Cobranzas"
        }]
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=4, ensure_ascii=False)
        return default_users
        
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_users(users):
    """Guarda la base de datos de usuarios en el archivo JSON."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def check_password_strength(password):
    """Evalúa la fortaleza de la contraseña."""
    score = 0
    if len(password) >= 8:
        score += 1
    if any(c.isupper() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(not c.isalnum() for c in password):
        score += 1
        
    configs = [
        {"label": "Muy débil ❌", "color": "#E53E3E"},
        {"label": "Débil ⚠️", "color": "#E53E3E"},
        {"label": "Regular ⚡", "color": "#D69E2E"},
        {"label": "Buena ✨", "color": "#3182CE"},
        {"label": "Muy segura ✅", "color": "#1B8C3E"}
    ]
    return configs[score]

def render_login():
    """Dibuja la pantalla de autenticación MiBanco."""
    # Inicializar estados de sesión para autenticación si no existen
    if 'auth_screen' not in st.session_state:
        st.session_state['auth_screen'] = 'login'
    if 'recovery_email' not in st.session_state:
        st.session_state['recovery_email'] = ''
    if 'recovery_code' not in st.session_state:
        st.session_state['recovery_code'] = ''
    if 'alert_message' not in st.session_state:
        st.session_state['alert_message'] = None
    if 'alert_type' not in st.session_state:
        st.session_state['alert_type'] = None
        
    users_db = load_users()
    
    # Inyectar estilos CSS para imitar fielmente la guía de estilos de MiBanco
    st.markdown("""
    <style>
        /* Importar fuente corporativa */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* Ocultar elementos nativos de Streamlit */
        [data-testid="stHeader"] {
            display: none !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Configuración del fondo con gradiente institucional verde */
        .stApp {
            background: linear-gradient(135deg, #1B8C3E 0%, #25A84E 40%, #4FC97A 70%, #7EDBA0 100%) !important;
            background-attachment: fixed !important;
        }
        
        /* Centrado general del layout */
        [data-testid="stAppViewContainer"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100vh !important;
            background: transparent !important;
        }
        
        [data-testid="stAppViewBlockContainer"] {
            max-width: 1000px !important;
            width: 100% !important;
            padding: 20px !important;
            margin: auto !important;
        }
        
        /* Tarjeta principal blanca redondeada compacta (Login Card) */
        div[data-testid="stVerticalBlockBorderContainer"]:has(.login-card-marker) {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1.5px solid #D5E8DC !important;
            border-radius: 16px !important;
            padding: 2.5rem !important; /* Margen interno cómodo de 2.5rem */
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.12) !important;
            max-width: 440px !important;
            width: 100% !important;
            margin: 0 auto !important;
            box-sizing: border-box !important;
        }
        
        /* Asegurar fondo blanco en los contenedores internos de la tarjeta */
        div[data-testid="stVerticalBlockBorderContainer"]:has(.login-card-marker) > div,
        div[data-testid="stVerticalBlockBorderContainer"]:has(.login-card-marker) [data-testid="stVerticalBlock"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
        }
        
        /* Forzar tipografía corporativa en toda la tarjeta */
        div[data-testid="stVerticalBlockBorderContainer"]:has(.login-card-marker) * {
            font-family: 'Outfit', -apple-system, sans-serif !important;
        }
        
        /* Inputs ocupan el 100% del contenedor y tienen bordes redondeados */
        div[data-testid="stTextInput"] input {
            border: 1.5px solid #D5E8DC !important;
            background-color: #FAFAFA !important;
            color: #1A3A2A !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #1B8C3E !important;
            box-shadow: 0 0 0 3px rgba(27, 140, 62, 0.15) !important;
            background-color: #FFFFFF !important;
            outline: none !important;
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #7A9088 !important;
            opacity: 0.7 !important;
        }
        
        /* Etiquetas de los inputs */
        div[data-testid="stTextInput"] label p {
            font-size: 13px !important;
            font-weight: 600 !important;
            color: #1A3A2A !important;
            margin-bottom: 4px !important;
        }
        
        /* Ajustes de checkbox */
        div[data-testid="stCheckbox"] label p {
            font-size: 13px !important;
            color: #7A9088 !important;
        }
        
        /* Botón enlace (Olvidé mi contraseña) */
        div:has(> .forgot-marker) + div button {
            background: none !important;
            border: none !important;
            color: #1B8C3E !important;
            text-decoration: underline !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            padding: 0 !important;
            cursor: pointer !important;
            box-shadow: none !important;
            width: auto !important;
            display: inline-block !important;
            margin: 0 !important;
            height: auto !important;
            min-height: unset !important;
            margin-top: 4px !important;
        }
        div:has(> .forgot-marker) + div button:hover {
            color: #167A36 !important;
            background: none !important;
            text-decoration: underline !important;
        }
        
        /* Botones primarios (Ingresar al sistema con gradiente institucional) */
        div:has(> .primary-btn-marker) + div button {
            background: linear-gradient(135deg, #1B8C3E 0%, #25A84E 40%, #4FC97A 70%, #7EDBA0 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            width: 100% !important;
            padding: 14px !important;
            box-shadow: 0 4px 10px rgba(27, 140, 62, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
            height: auto !important;
            display: block !important;
            cursor: pointer !important;
        }
        div:has(> .primary-btn-marker) + div button:hover {
            background: linear-gradient(135deg, #167A36 0%, #1B8C3E 40%, #25A84E 70%, #4FC97A 100%) !important;
            box-shadow: 0 6px 20px rgba(27, 140, 62, 0.35) !important;
            transform: translateY(-1px) !important;
        }
        div:has(> .primary-btn-marker) + div button:active {
            transform: translateY(1px) !important;
        }
        
        /* Botones secundarios */
        div:has(> .secondary-btn-marker) + div button {
            background-color: transparent !important;
            color: #1B8C3E !important;
            border: 1.5px solid #1B8C3E !important;
            border-radius: 10px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            width: 100% !important;
            padding: 12px !important;
            transition: all 0.2s ease-in-out !important;
            height: auto !important;
            display: block !important;
            cursor: pointer !important;
        }
        div:has(> .secondary-btn-marker) + div button:hover {
            background-color: #F0F9F4 !important;
            color: #167A36 !important;
            border-color: #167A36 !important;
        }
        
        /* Divisor visual decorativo */
        .auth-divider {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 20px 0;
            color: #B0C8BC;
            font-size: 12px;
            font-weight: 600;
        }
        .auth-divider::before, .auth-divider::after {
            content: "";
            flex: 1;
            height: 1px;
            background: #E8F4EC;
        }
        
        /* Alertas de autenticación */
        .auth-alert {
            border-radius: 10px;
            padding: 12px 16px;
            font-size: 13px;
            margin-bottom: 18px;
            line-height: 1.4;
            border: 1px solid transparent;
            font-weight: 500;
        }
        .auth-alert-success { background-color: #F0FFF4; border-color: #9AE6B4; color: #276749; }
        .auth-alert-error { background-color: #FFF5F5; border-color: #FEB2B2; color: #C53030; }
        .auth-alert-info { background-color: #EBF8FF; border-color: #90CDF4; color: #2C5282; }
    </style>
    """, unsafe_allow_html=True)
    
    # 1. Estructura de 3 columnas para centrado y equilibrio perfecto de la tarjeta
    col_left, col_center, col_right = st.columns([1, 2.2, 1])
    
    with col_center:
        # Contenedor principal de la tarjeta de login
        with st.container(border=True):
            st.markdown('<div class="login-card-marker"></div>', unsafe_allow_html=True)
            
            # 3. Logotipo oficial MiBanco integrado al contenedor blanco
            st.markdown(
                '<div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 24px;">'
                '    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 8px;">'
                '        <svg width="40" height="40" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">'
                '            <g transform="translate(40,40)">'
                '                <!-- Rayos del Sol (Amarillo brillante #FDD700) -->'
                '                <circle cx="0" cy="0" r="14" fill="#FDD700"/>'
                '                <g fill="#FDD700">'
                '                    <rect x="-3" y="-30" width="6" height="10" rx="2.5"/>'
                '                    <rect x="-3" y="20" width="6" height="10" rx="2.5"/>'
                '                    <rect x="20" y="-3" width="10" height="6" rx="2.5"/>'
                '                    <rect x="-30" y="-3" width="10" height="6" rx="2.5"/>'
                '                    <rect x="10" y="-25" width="6" height="10" rx="2.5" transform="rotate(45,13,-20)"/>'
                '                    <rect x="-25" y="-25" width="6" height="10" rx="2.5" transform="rotate(-45,-22,-20)"/>'
                '                    <rect x="10" y="15" width="6" height="10" rx="2.5" transform="rotate(-45,13,20)"/>'
                '                    <rect x="-25" y="15" width="6" height="10" rx="2.5" transform="rotate(45,-22,20)"/>'
                '                </g>'
                '            </g>'
                '        </svg>'
                '        <!-- Tipografía mibanco bold estilizada (Verde pino #1A3A2A) -->'
                '        <span style="font-family: \'Outfit\', sans-serif; font-size: 32px; font-weight: 800; color: #1A3A2A; letter-spacing: -1.2px; line-height: 1;">mibanco</span>'
                '    </div>'
                '    <span style="font-size: 11px; color: #1B8C3E; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; background-color: #F0F9F4; border: 1px solid #C8EDD8; border-radius: 20px; padding: 4px 14px; text-align: center; display: inline-block;">'
                '        🔒 Sistema CobraIQ · Solo autorizado'
                '    </span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Mostrar notificaciones o alertas del sistema si existen
            if st.session_state['alert_message']:
                atype = st.session_state['alert_type']
                st.markdown(f'<div class="auth-alert auth-alert-{atype}">{st.session_state["alert_message"]}</div>', unsafe_allow_html=True)
                st.session_state['alert_message'] = None
                st.session_state['alert_type'] = None
                
            screen = st.session_state['auth_screen']
            
            # ───────────────────────────────────────────────────────────────────
            # PANTALLA 1: INICIAR SESIÓN (LOGIN)
            # ───────────────────────────────────────────────────────────────────
            if screen == 'login':
                # Títulos en verde pino (#1A3A2A) y subtítulo en verde secundario (#7A9088)
                st.markdown("<div style='font-size: 20px; font-weight: 700; color: #1A3A2A; text-align: center; margin-bottom: 6px;'>Ingresa a tu cuenta</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 13px; color: #7A9088; text-align: center; margin-bottom: 25px;'>Acceso exclusivo para colaboradores Mibanco</div>", unsafe_allow_html=True)
                
                email = st.text_input("Correo corporativo", placeholder="trabajador@mibanco.com.pe", key="email_login")
                password = st.text_input("Contraseña", type="password", placeholder="••••••••", key="pass_login")
                
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                col_rem, col_forg = st.columns([1.1, 0.9])
                with col_rem:
                    remember = st.checkbox("Recordarme", value=True, key="rem_login")
                with col_forg:
                    st.markdown('<div class="forgot-marker"></div>', unsafe_allow_html=True)
                    if st.button("Olvidé mi contraseña", key="btn_to_forgot", use_container_width=True):
                        st.session_state['auth_screen'] = 'forgot'
                        st.rerun()
                        
                st.markdown('<div class="primary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("Ingresar al sistema", type="primary", key="btn_do_login", use_container_width=True):
                    email_clean = email.strip().lower()
                    if not email_clean:
                        st.session_state['alert_message'] = "⚠️ Ingresa tu correo corporativo."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif not password:
                        st.session_state['alert_message'] = "⚠️ Ingresa tu contraseña."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    else:
                        # Validar credenciales
                        user = next((u for u in users_db if u['email'].lower() == email_clean and u['password'] == password), None)
                        if user:
                            st.session_state['username'] = user['nombre']
                            st.session_state['user_email'] = user['email']
                            st.session_state['user_area'] = user['area']
                            st.session_state['auth_screen'] = 'welcome'
                            st.rerun()
                        else:
                            st.session_state['alert_message'] = "❌ Correo o contraseña incorrectos. Verifica tus datos o recupera tu contraseña."
                            st.session_state['alert_type'] = "error"
                            st.rerun()
                            
                st.markdown('<div class="auth-divider">o</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="secondary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("➕ Crear cuenta de colaborador", type="secondary", key="btn_to_register", use_container_width=True):
                    st.session_state['auth_screen'] = 'register'
                    st.rerun()
                    
                st.markdown(
                    '<div style="text-align: center; margin-top: 22px; font-size: 13px; color: #7A9088;">'
                    '  ¿Problemas para ingresar? Contacta a <strong>soporte@mibanco.com.pe</strong>'
                    '</div>',
                    unsafe_allow_html=True
                )
                
            # ───────────────────────────────────────────────────────────────────
            # PANTALLA 2: CREAR CUENTA (REGISTER)
            # ───────────────────────────────────────────────────────────────────
            elif screen == 'register':
                st.markdown("<div style='font-size: 20px; font-weight: 700; color: #1A3A2A; text-align: center; margin-bottom: 6px;'>Crear cuenta de colaborador</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 13px; color: #7A9088; text-align: center; margin-bottom: 25px;'>Completa tus datos para registrarte en CobraIQ</div>", unsafe_allow_html=True)
                
                nombre = st.text_input("Nombre completo", placeholder="Ej: María García López", key="reg_nombre")
                area = st.text_input("Área / Cargo", placeholder="Ej: Gestión de Cobranzas", key="reg_area")
                email = st.text_input("Correo corporativo", placeholder="tunombre@mibanco.com.pe", key="reg_email")
                
                password = st.text_input("Contraseña", type="password", placeholder="Mínimo 8 caracteres", key="reg_pass")
                if password:
                    strength = check_password_strength(password)
                    st.markdown(
                        f'<div style="font-size: 12px; margin-top: -8px; margin-bottom: 8px; font-weight: 600; color: {strength["color"]};">'
                        f'    Fortaleza de contraseña: {strength["label"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                password_confirm = st.text_input("Confirmar contraseña", type="password", placeholder="Repite tu contraseña", key="reg_pass2")
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="primary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("✅ Crear mi cuenta", type="primary", key="btn_do_register", use_container_width=True):
                    email_clean = email.strip().lower()
                    if not nombre:
                        st.session_state['alert_message'] = "⚠️ Ingresa tu nombre completo."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif not area:
                        st.session_state['alert_message'] = "⚠️ Ingresa tu área o cargo."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif not email_clean:
                        st.session_state['alert_message'] = "⚠️ Ingresa tu correo corporativo."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif len(password) < 8:
                        st.session_state['alert_message'] = "⚠️ La contraseña debe tener al menos 8 caracteres."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif password != password_confirm:
                        st.session_state['alert_message'] = "❌ Las contraseñas no coinciden."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif any(u for u in users_db if u['email'].lower() == email_clean):
                        st.session_state['alert_message'] = "❌ Este correo ya está registrado en el sistema."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    else:
                        new_user = {
                            "email": email_clean,
                            "password": password,
                            "nombre": nombre,
                            "area": area
                        }
                        users_db.append(new_user)
                        save_users(users_db)
                        
                        st.session_state['alert_message'] = f"✅ Cuenta creada exitosamente para {nombre}. Ya puedes iniciar sesión."
                        st.session_state['alert_type'] = "success"
                        st.session_state['auth_screen'] = 'login'
                        st.rerun()
                        
                st.markdown('<div class="secondary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("← Volver al inicio", type="secondary", key="btn_reg_back", use_container_width=True):
                    st.session_state['auth_screen'] = 'login'
                    st.rerun()
                    
            # ───────────────────────────────────────────────────────────────────
            # PANTALLA 3: RECUPERAR CONTRASEÑA (FORGOT)
            # ───────────────────────────────────────────────────────────────────
            elif screen == 'forgot':
                st.markdown("<div style='font-size: 20px; font-weight: 700; color: #1A3A2A; text-align: center; margin-bottom: 6px;'>Recuperar contraseña</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 13px; color: #7A9088; text-align: center; margin-bottom: 25px;'>Ingresa tu correo y te enviaremos un código de verificación</div>", unsafe_allow_html=True)
                
                email = st.text_input("Correo corporativo registrado", placeholder="tunombre@mibanco.com.pe", key="forgot_email")
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="primary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("📧 Enviar código de recuperación", type="primary", key="btn_do_forgot", use_container_width=True):
                    email_clean = email.strip().lower()
                    user = next((u for u in users_db if u['email'].lower() == email_clean), None)
                    if not user:
                        st.session_state['alert_message'] = "❌ No encontramos una cuenta registrada con ese correo electrónico."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    else:
                        code = str(random.randint(100000, 999999))
                        st.session_state['recovery_code'] = code
                        st.session_state['recovery_email'] = email_clean
                        st.session_state['auth_screen'] = 'code'
                        st.rerun()
                        
                st.markdown('<div class="secondary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("← Volver al inicio", type="secondary", key="btn_forgot_back", use_container_width=True):
                    st.session_state['auth_screen'] = 'login'
                    st.rerun()
                    
            # ───────────────────────────────────────────────────────────────────
            # PANTALLA 4: INGRESAR CÓDIGO (CODE)
            # ───────────────────────────────────────────────────────────────────
            elif screen == 'code':
                st.markdown("<div style='font-size: 20px; font-weight: 700; color: #1A3A2A; text-align: center; margin-bottom: 6px;'>Verificar código</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 13px; color: #7A9088; text-align: center; margin-bottom: 25px;'>Ingresa el código de 6 dígitos enviado a tu correo corporativo.</div>", unsafe_allow_html=True)
                
                demo_code = st.session_state['recovery_code']
                st.markdown(
                    f'<div class="auth-alert auth-alert-info">'
                    f'    💡 Código demo: <strong>{demo_code}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                code_input = st.text_input("Código de verificación", placeholder="000000", max_chars=6, key="code_val")
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="primary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("✅ Verificar código", type="primary", key="btn_do_code", use_container_width=True):
                    if code_input.strip() != demo_code:
                        st.session_state['alert_message'] = "❌ Código incorrecto. Inténtalo de nuevo."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    else:
                        st.session_state['auth_screen'] = 'newpass'
                        st.rerun()
                        
                st.markdown('<div class="secondary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("← Reenviar código", type="secondary", key="btn_code_back", use_container_width=True):
                    st.session_state['auth_screen'] = 'forgot'
                    st.rerun()
                    
            # ───────────────────────────────────────────────────────────────────
            # PANTALLA 5: NUEVA CONTRASEÑA (NEWPASS)
            # ───────────────────────────────────────────────────────────────────
            elif screen == 'newpass':
                st.markdown("<div style='font-size: 20px; font-weight: 700; color: #1A3A2A; text-align: center; margin-bottom: 6px;'>Nueva contraseña</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 13px; color: #7A9088; text-align: center; margin-bottom: 25px;'>Escribe tu nueva contraseña para CobraIQ</div>", unsafe_allow_html=True)
                
                new_pass = st.text_input("Nueva contraseña", type="password", placeholder="Mínimo 8 caracteres", key="new_password")
                if new_pass:
                    strength = check_password_strength(new_pass)
                    st.markdown(
                        f'<div style="font-size: 12px; margin-top: -8px; margin-bottom: 8px; font-weight: 600; color: {strength["color"]};">'
                        f'    Fortaleza de contraseña: {strength["label"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                new_pass_confirm = st.text_input("Confirmar nueva contraseña", type="password", placeholder="Repite la contraseña", key="new_password_confirm")
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="primary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("🔒 Guardar nueva contraseña", type="primary", key="btn_do_newpass", use_container_width=True):
                    if len(new_pass) < 8:
                        st.session_state['alert_message'] = "⚠️ La contraseña debe tener al menos 8 caracteres."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    elif new_pass != new_pass_confirm:
                        st.session_state['alert_message'] = "❌ Las contraseñas no coinciden."
                        st.session_state['alert_type'] = "error"
                        st.rerun()
                    else:
                        # Actualizar en la base de datos
                        email = st.session_state['recovery_email']
                        for u in users_db:
                            if u['email'].lower() == email.lower():
                                u['password'] = new_pass
                                break
                        save_users(users_db)
                        
                        st.session_state['alert_message'] = "✅ Contraseña actualizada. Ya puedes iniciar sesión."
                        st.session_state['alert_type'] = "success"
                        st.session_state['auth_screen'] = 'login'
                        st.session_state['recovery_code'] = ''
                        st.session_state['recovery_email'] = ''
                        st.rerun()
                        
            # ───────────────────────────────────────────────────────────────────
            # PANTALLA 6: BIENVENIDA (WELCOME)
            # ───────────────────────────────────────────────────────────────────
            elif screen == 'welcome':
                # Obtener iniciales
                name = st.session_state.get('username', 'Usuario')
                area = st.session_state.get('user_area', 'CobraIQ')
                email = st.session_state.get('user_email', '')
                initials = "".join([part[0] for part in name.split()[:2]]).upper()
                
                st.markdown(f"""<div style="text-align: center; padding: 20px 0;">
<div style="width: 72px; height: 72px; border-radius: 50%; background: linear-gradient(135deg, #1B8C3E, #4FC97A); color: #ffffff; font-size: 26px; font-weight: 700; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; border: 2px solid #D5E8DC; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">{initials}</div>
<div style="font-size: 20px; font-weight: 700; color: #1A3A2A; margin-bottom: 6px;">¡Bienvenido/a, {name.split()[0]}!</div>
<div style="font-size: 13px; color: #7A9088; margin-bottom: 4px;">{area}</div>
<div style="font-size: 12px; color: #B0C8BC; margin-bottom: 24px;">{email}</div>
<div style="background-color: #F0FFF4; border: 1px solid #9AE6B4; border-radius: 12px; padding: 16px; margin-bottom: 20px; text-align: left;">
<div style="font-size: 13px; color: #276749; font-weight: 600; margin-bottom: 6px;">✅ Acceso autorizado a CobraIQ</div>
<div style="font-size: 12px; color: #3A7A52; line-height: 1.4;">Tu sesión está activa. Presiona el botón de abajo para ingresar al Dashboard.</div>
</div>
</div>""", unsafe_allow_html=True)
                
                st.markdown('<div class="primary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("🚀 Ir al Dashboard CobraIQ ahora", type="primary", key="btn_welcome_go", use_container_width=True):
                    # Marcar sesión activa
                    st.session_state['authenticated'] = True
                    st.rerun()
                    
                st.markdown('<div class="secondary-btn-marker"></div>', unsafe_allow_html=True)
                if st.button("Cerrar sesión", type="secondary", key="btn_welcome_logout", use_container_width=True):
                    # Limpiar variables de sesión y volver
                    st.session_state.pop('username', None)
                    st.session_state.pop('user_email', None)
                    st.session_state.pop('user_area', None)
                    st.session_state['authenticated'] = False
                    st.session_state['auth_screen'] = 'login'
                    st.rerun()
