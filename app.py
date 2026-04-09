from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import joblib
import pandas as pd
import warnings

# Silenciamos la advertencia de los nombres de las columnas para tener una consola limpia
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

#### CONFIGURACION DE SQLALCHEMY ####
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/dbG6_proyecto_malaria'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

#### CARGA DE MODELOS ML ####
PATH_MODELO = './modelo_malaria_rf.pkl'
PATH_TRANSFORMER = './transformer_malaria.pkl'

modelo_rf = joblib.load(PATH_MODELO)
transformer = joblib.load(PATH_TRANSFORMER)

### CREAMOS LA CLASE QUE VA A CONVERTIRSE EN UNA TABLA SQL
class Prediccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    departamento = db.Column(db.String(100), nullable=False)
    provincia = db.Column(db.String(100), nullable=False)
    ano = db.Column(db.Integer, nullable=False)
    semana = db.Column(db.Integer, nullable=False)
    casos_predichos = db.Column(db.Double, nullable=False)

# Esquema para Marshmallow (para serializar respuestas JSON)
class PrediccionSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Prediccion

prediccion_schema = PrediccionSchema()
predicciones_schema = PrediccionSchema(many=True)

#### FUNCION PREDICTIVA ####
def predecir_casos(departamento, provincia, ano, semana):
    nuevo_dato = pd.DataFrame([{
        'departamento': departamento.upper(), # Aseguramos mayúsculas como en el dataset
        'provincia': provincia.upper(),
        'ano': int(ano),
        'semana': int(semana),
        'total_casos': 0 
    }])
    
    dato_transformado = transformer.transform(nuevo_dato)
    
    if isinstance(dato_transformado, pd.DataFrame):
        cols_a_borrar = [col for col in dato_transformado.columns if 'total_casos' in col]
        dato_transformado = dato_transformado.drop(columns=cols_a_borrar)
    else:
        dato_transformado = dato_transformado[:, :-1]
        
    prediccion = modelo_rf.predict(dato_transformado)
    return float(prediccion[0])

#### RUTAS DE LA APLICACIÓN ####

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.json
    
    # Extraemos los datos enviados desde el frontend
    dept = data.get('departamento')
    prov = data.get('provincia')
    anio = data.get('ano')
    sem = data.get('semana')
    
    # Llamamos a nuestra función de Machine Learning
    casos_estimados = predecir_casos(dept, prov, anio, sem)
    
    # Guardamos la consulta y el resultado en la base de datos MySQL
    nueva_consulta = Prediccion(
        departamento=dept.upper(),
        provincia=prov.upper(),
        ano=anio,
        semana=sem,
        casos_predichos=casos_estimados
    )
    
    db.session.add(nueva_consulta)
    db.session.commit()
    
    # Retornamos el registro guardado en formato JSON
    return jsonify(prediccion_schema.dump(nueva_consulta)), 200

@app.route('/registros', methods=['GET'])
def get_all_records():
    # Obtiene todo el historial de predicciones
    all_records = Prediccion.query.all()
    return jsonify(predicciones_schema.dump(all_records)), 200

@app.route('/registro/<int:id>', methods=['DELETE'])
def delete_record(id):
    # Elimina un registro por su ID
    registro = Prediccion.query.get(id)
    if not registro:
        return jsonify({"mensaje": "Registro no encontrado"}), 404
        
    db.session.delete(registro)
    db.session.commit()
    return jsonify({"mensaje": "Registro eliminado exitosamente"}), 200

if __name__ == '__main__':
    # Crea las tablas en la base de datos si no existen
    with app.app_context():
        db.create_all()
    app.run(debug=True)