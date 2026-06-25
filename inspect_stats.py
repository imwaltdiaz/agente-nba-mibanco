import pandas as pd

# Clientes
df_c = pd.read_csv('data/01_raw/01_Tabla_de_Clientes.csv')
print('=== CLIENTES ===')
print('Shape:', df_c.shape)
for col in ['score_riesgo','prob_default','es_digital','uso_app','ratio_pago']:
    if col in df_c.columns:
        pn = df_c[col].isnull().mean()*100
        print(f'  {col}: {pn:.1f}% nulos')

# Creditos
df_cr = pd.read_csv('data/01_raw/02_Tabla_de_Crditos.csv')
print()
print('=== CREDITOS ===')
print('Shape:', df_cr.shape)
periodos = sorted(df_cr['periodo'].astype(str).unique())
print('Periodos:', periodos)
print('Clientes unicos:', df_cr['cliente_id'].nunique())
for col in ['dias_mora','saldo_restante','cuota_mensual','pago_realizado_mes']:
    if col in df_cr.columns:
        pn = df_cr[col].isnull().mean()*100
        print(f'  {col}: {pn:.1f}% nulos')

# Contactos
df_ct = pd.read_csv('data/01_raw/03_Tabla_contactos.csv', sep=';')
print()
print('=== CONTACTOS ===')
print('Shape:', df_ct.shape)
df_ct['fecha_dt'] = pd.to_datetime(df_ct['fecha_contacto'], dayfirst=True, errors='coerce')
print('Fecha min:', df_ct['fecha_dt'].min().date())
print('Fecha max:', df_ct['fecha_dt'].max().date())
print('Clientes unicos:', df_ct['cliente_id'].nunique())
for col in ['costo_contacto','pago_7d_post_contacto','canal_contacto']:
    pn = df_ct[col].isnull().mean()*100
    print(f'  {col}: {pn:.1f}% nulos')
print('Tasa pago global:', round(df_ct['pago_7d_post_contacto'].mean(), 4))
print('Canales:')
print(df_ct['canal_contacto'].value_counts().to_string())
print()
print('Contactos por mes:')
df_ct['mes'] = df_ct['fecha_dt'].dt.to_period('M').astype(str)
print(df_ct['mes'].value_counts().sort_index().to_string())
