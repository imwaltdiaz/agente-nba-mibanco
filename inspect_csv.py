#!/usr/bin/env python
"""Script para inspeccionar estructura del CSV de contactos"""
import pandas as pd

# Leer primeras filas del CSV de contactos
df = pd.read_csv('data/01_raw/03_Tabla_contactos.csv', nrows=5)
print("COLUMNAS DEL CSV DE CONTACTOS:")
print(df.columns.tolist())
print("\nPRIMERAS 2 FILAS:")
print(df.head(2))
print("\nTIPOS DE DATOS:")
print(df.dtypes)
