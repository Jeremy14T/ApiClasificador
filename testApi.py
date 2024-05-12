import requests

# URL de la API
url = 'http://127.0.0.1:5000/classify_cv'

# Nombre del archivo PDF que deseas enviar
file_path = 'CV1.pdf'

# Datos adicionales para enviar en la solicitud POST
data = {
    'categoria': 'Dotnet Developer',
    'limpiador_path': 'diccionarios'
}

# Abrir el archivo PDF y prepararlo para enviar como un archivo adjunto
with open(file_path, 'rb') as file:
    files = {'file': (file_path, file, 'application/pdf')}  # 'file' es el nombre del campo en el formulario

    # Enviar la solicitud POST a la API
    response = requests.post(url, files=files, data=data)

# Obtener los resultados de la API
if response.status_code == 200:
    predictions = response.json()
    for prediction in predictions:
        file_path = prediction['file_path']
        predicted_category = prediction.get('predicted_category', 'Error: No se pudo predecir la categoría')
        print(f"Archivo: {file_path}, Categoría predicha: {predicted_category}")
else:
    print("Error al hacer la solicitud a la API:", response.text)
