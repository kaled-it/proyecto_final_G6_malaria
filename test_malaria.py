import joblib
import pandas as pd
import numpy as np

# Rutas de los archivos exportados en el notebook anterior
# (Asegúrate de ajustar las rutas a relativas como './modelo_malaria_rf.pkl' 
# si mueves los archivos a la carpeta de tu proyecto local en VS Code)
PATH_MODELO = 'C:\\Users\\icale\\Documents\\CODIGO\\Codigo_proyecto_final\\Codigo_malaria\\modelo_malaria_rf.pkl'
PATH_TRANSFORMER = 'C:\\Users\\icale\\Documents\\CODIGO\\Codigo_proyecto_final\\Codigo_malaria\\transformer_malaria.pkl'

# Cargar el modelo y el transformer
modelo_rf = joblib.load(PATH_MODELO)
transformer = joblib.load(PATH_TRANSFORMER)

print("¡Modelo y transformer cargados exitosamente!")

def predict_malaria_cases(
    departamento: str,
    provincia: str,
    ano: int,
    semana: int
) -> float:
    """
    Predice el número de casos de malaria basado en nuevos datos de entrada.
    """

    # 1. Crear el DataFrame agregando 'total_casos': 0 para cumplir 
    # con la estructura que el transformer memorizó en el entrenamiento.
    nuevo_dato = pd.DataFrame([{
        'departamento': departamento,
        'provincia': provincia,
        'ano': ano,
        'semana': semana,
        'total_casos': 0  # <--- Dato ficticio para el transformer
    }])

    # 2. Aplicar el transformer
    dato_transformado = transformer.transform(nuevo_dato)

    # 3. Retirar la columna 'total_casos' antes de predecir, ya que el 
    # modelo Random Forest fue entrenado estrictamente sin esa columna.
    if isinstance(dato_transformado, pd.DataFrame):
        # Si scikit-learn devuelve un DataFrame, buscamos y eliminamos la columna
        cols_a_borrar = [col for col in dato_transformado.columns if 'total_casos' in col]
        dato_transformado = dato_transformado.drop(columns=cols_a_borrar)
    else:
        # Si devuelve un arreglo de Numpy, quitamos la última columna (que es total_casos)
        dato_transformado = dato_transformado[:, :-1]

    # 4. Realizar la predicción usando el modelo cargado
    prediccion = modelo_rf.predict(dato_transformado)

    return prediccion[0]
# ==========================================
# Ejemplo de uso (Testing)
# ==========================================
if __name__ == "__main__":
    
    # Parámetros de prueba
    nuevos_datos = {
        'departamento': 'LORETO',
        'provincia': 'MAYNAS',
        'ano': 2024,
        'semana': 15
    }

    # Llamada a la función predictiva
    prediccion_casos = predict_malaria_cases(**nuevos_datos)
    
    print("-" * 50)
    print("RESULTADO DE LA PREDICCIÓN")
    print("-" * 50)
    print(f"Departamento: {nuevos_datos['departamento']}")
    print(f"Provincia: {nuevos_datos['provincia']}")
    print(f"Año: {nuevos_datos['ano']} | Semana Epidemiológica: {nuevos_datos['semana']}")
    print(f"-> PICO PREDICHO: {prediccion_casos:.2f} casos")
    print("-" * 50)