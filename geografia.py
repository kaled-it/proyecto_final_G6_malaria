import pandas as pd
from sqlalchemy import create_engine

print("Cargando el archivo...")
df = pd.read_csv('vigilancia_malaria_2009_2024_cleaned.csv', low_memory=False)

print("Procesando la jerarquía geográfica completa...")
directorio = df[['departamento', 'provincia', 'distrito', 'localidad']].drop_duplicates()
directorio = directorio.dropna()

directorio = directorio.rename(columns={'departamento': 'region'})

print(f"¡Listo! Se encontraron {len(directorio)} rutas únicas.")

print("Conectando a MySQL y subiendo datos...")
engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/dbG6_proyecto_malaria')

directorio.to_sql('geografia', con=engine, if_exists='replace', index=False)

print("¡Proceso terminado! La tabla 'geografia' ahora tiene región, provincia, distrito y localidad.")