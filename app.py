from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy import text
import joblib
import pandas as pd
import warnings

# Silenciamos advertencias de nombres de columnas
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

#### CONFIGURACION DE SQLALCHEMY ####
app.app_context().push()
# Asegúrate de que el nombre de la BD sea el correcto (dbG6_proyecto_malaria)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/dbG6_proyecto_malaria'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

#### CARGA DE MODELOS ML ####
PATH_MODELO = './modelos/modelo_malaria_rf.pkl'
PATH_TRANSFORMER = './modelos/transformer_malaria.pkl'

modelo_rf = joblib.load(PATH_MODELO)
transformer = joblib.load(PATH_TRANSFORMER)

#### MODELOS DE BASE DE DATOS ####

class Prediccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    departamento = db.Column(db.String(100), nullable=False)
    provincia = db.Column(db.String(100), nullable=False)
    ano = db.Column(db.Integer, nullable=False)
    semana = db.Column(db.Integer, nullable=False)
    casos_predichos = db.Column(db.Double, nullable=False)

class Geografia(db.Model):
    __tablename__ = 'geografia'
    # Si usaste to_sql de pandas, es probable que no haya 'id'. 
    # Definimos 'localidad' o la combinación como llave si no tienes un ID autoincremental.
    region = db.Column(db.String(100), primary_key=True) # Definimos temporalmente como PK para consulta
    provincia = db.Column(db.String(100), primary_key=True)
    distrito = db.Column(db.String(100), primary_key=True)
    localidad = db.Column(db.String(100), primary_key=True)

# Esquemas de Serialización
class PrediccionSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Prediccion

prediccion_schema = PrediccionSchema()
predicciones_schema = PrediccionSchema(many=True)

#### FUNCION PREDICTIVA ####
def predecir_casos(region, provincia, ano, semana):
    # IMPORTANTE: El transformer espera 'departamento', no 'region'
    nuevo_dato = pd.DataFrame([{
        'departamento': region.upper(), 
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
    
    # Extraemos con los nombres que enviará el nuevo HTML
    reg = data.get('region') 
    prov = data.get('provincia')
    anio = data.get('ano')
    sem = data.get('semana')
    
    # Predicción
    casos_estimados = predecir_casos(reg, prov, anio, sem)
    
    # Guardamos en el historial (usando nombres de la tabla Prediccion)
    nueva_consulta = Prediccion(
        departamento=reg.upper(),
        provincia=prov.upper(),
        ano=anio,
        semana=sem,
        casos_predichos=casos_estimados
    )
    
    db.session.add(nueva_consulta)
    db.session.commit()
    
    return jsonify(prediccion_schema.dump(nueva_consulta)), 200

@app.route('/registros', methods=['GET'])
def get_all_records():
    all_records = Prediccion.query.order_by(Prediccion.id.desc()).all()
    return jsonify(predicciones_schema.dump(all_records)), 200

@app.route('/registro/<int:id>', methods=['DELETE'])
def delete_record(id):
    registro = Prediccion.query.get(id)
    if not registro:
        return jsonify({"mensaje": "Registro no encontrado"}), 404
    db.session.delete(registro)
    db.session.commit()
    return jsonify({"mensaje": "Registro eliminado exitosamente"}), 200

#### ENDPOINTS PARA MENÚS DESPLEGABLES ####

@app.route('/api/regiones', methods=['GET'])
def get_regiones():
    regiones = db.session.query(Geografia.region).distinct().all()
    return jsonify(sorted([r[0] for r in regiones]))

@app.route('/api/provincias', methods=['GET'])
def get_provincias():
    region_sel = request.args.get('region')
    provincias = db.session.query(Geografia.provincia).filter_by(region=region_sel).distinct().all()
    return jsonify(sorted([p[0] for p in provincias]))

@app.route('/api/distritos', methods=['GET'])
def get_distritos():
    region_sel = request.args.get('region')
    prov_sel = request.args.get('provincia')
    distritos = db.session.query(Geografia.distrito).filter_by(region=region_sel, provincia=prov_sel).distinct().all()
    return jsonify(sorted([d[0] for d in distritos]))

@app.route('/api/localidades', methods=['GET'])
def get_localidades():
    region_sel = request.args.get('region')
    prov_sel = request.args.get('provincia')
    dist_sel = request.args.get('distrito')
    localidades = db.session.query(Geografia.localidad).filter_by(region=region_sel, provincia=prov_sel, distrito=dist_sel).distinct().all()
    return jsonify(sorted([l[0] for l in localidades]))

if __name__ == '__main__':
    with app.app_context():
        # db.create_all() # Cuidado: Si 'geografia' ya existe por pandas, no la sobreescribirá.
        pass
    app.run(debug=True)