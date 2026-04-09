import pandas as pd
from sqlalchemy import create_engine

# 1. Cargar el archivo original 
print("Cargando el archivo...")
df = pd.read_csv('vigilancia_malaria_2009_2024_cleaned.csv', low_memory=False)

# 2. Extraer usando los nombres reales y eliminar repeticiones
print("Procesando combinaciones únicas...")
directorio = df[['departamento', 'localidad']].drop_duplicates()

# Limpiamos valores vacíos
directorio = directorio.dropna()

# 3. EL TRUCO: Renombrar 'departamento' a 'region' para que MySQL lo acepte
directorio = directorio.rename(columns={'departamento': 'region'})

print(f"¡Listo! Se encontraron {len(directorio)} combinaciones únicas.")

# 4. Conectarnos a la base de datos y subir los datos
print("Conectando a MySQL y subiendo datos...")
engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/dbG6_proyecto_malaria')

# Subimos el dataframe directo a la tabla 'geografia'
directorio.to_sql('geografia', con=engine, if_exists='append', index=False)

print("¡Proceso terminado con éxito! Tu base de datos ya tiene el directorio completo.")