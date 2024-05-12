import os
from flask import Flask, request, jsonify
from Modelo import ModeloClasificador
from limpieza_unique import LimpiadorCV
import PyPDF2
from werkzeug.utils import secure_filename
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  #

# Asegúrate de tener este directorio creado o maneja la creación dinámicamente
UPLOAD_FOLDER = 'Pruebas_con_PDFs'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clasificar_cv(uploaded_file, categoria, limpiador):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Aplicar limpieza del currículum
        clean_resume = limpiador.clean_resume(text)
        clean_resume = limpiador.clean_useless_words(clean_resume, categoria)

        clasificador = ModeloClasificador('svm_model.joblib', 'tfidf_vectorizer.joblib', 'selected_features.joblib', 'pca.joblib', 'label_encoder.joblib')
        predicted_category = clasificador.predecir(clean_resume)

        # Si predicted_category es un ndarray, convertir a un tipo adecuado:
        if isinstance(predicted_category, np.ndarray):
            predicted_category = predicted_category.tolist()  # Convertir a lista si es un arreglo
            if len(predicted_category) == 1:
                predicted_category = predicted_category[0]  # Tomar el primer elemento si solo hay uno

        return predicted_category
    
    except FileNotFoundError:
        return f"Error: Archivo PDF no encontrado."
    except Exception as e:
        return f"Error inesperado: {str(e)}"

def categoria_valida(categoria):
    # Obtener la ruta del directorio de diccionarios relativa al directorio actual
    diccionarios_dir = os.path.join(os.path.dirname(__file__), 'diccionarios')
    # Listar archivos de texto en el directorio
    archivos_txt = [f[:-4] for f in os.listdir(diccionarios_dir) if f.endswith('.txt')]
    # Verificar si la categoría está en la lista de nombres de archivo
    return categoria in archivos_txt

@app.route('/clasificar', methods=['POST'])
def clasificar():
    print('Categoría recibida:', request.form['categoria'])

    # Mostrar los archivos recibidos en la solicitud
    for file in request.files.getlist('file'):
        print('Archivo recibido:', file.filename)
    print("--------------------------")    
    # Verificar si se ha enviado un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha enviado ningún archivo.'}), 400

    uploaded_files = request.files.getlist('file')
    categoria = request.form.get('categoria')  # Obtener la categoría del formulario
    if not categoria:  # Si la categoría no se proporciona
        return jsonify({'error': 'No se seleccionó ninguna categoría. Favor de seleccionar una categoría.'}), 400

    limpiador = LimpiadorCV("diccionarios")

    if not categoria_valida(categoria):
        return jsonify({'error': f'Categoría "{categoria}" no válida. Favor de seleccionar una categoría existente.'}), 400

    resultados = []

    archivos_con_categoria = []  # Almacenar nombres de archivos con la categoría seleccionada

    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.filename):
            # Clasificar el currículum
            categoria_predicha = clasificar_cv(uploaded_file, categoria, limpiador)
            resultados.append({'nombre_archivo': uploaded_file.filename, 'categoria_predicha': categoria_predicha})

            # Si la categoría predicha coincide con la categoría seleccionada, añadir el nombre del archivo a la lista
            if categoria_predicha == categoria:
                archivos_con_categoria.append(uploaded_file.filename)
        else:
            resultados.append({'error': 'Tipo de archivo no permitido o no se proporcionó ningún archivo.'})

    # Agregar los nombres de archivos que coinciden con la categoría seleccionada al resultado
    resultados.append({'archivos_con_categoria': archivos_con_categoria})

    return jsonify(resultados)

if __name__ == "__main__":
    app.run(debug=True)