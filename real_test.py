from Modelo import ModeloClasificador
from limpieza_unique import LimpiadorCV
import PyPDF2

def clasificar_cv(file_paths, categoria, limpiador):
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''.join([page.extract_text() for page in reader.pages])

            #print(f"\nLeyendo archivo: {file_path}")
            #print(f"Longitud del texto extraído: {len(text)}")

            clean_resume = limpiador.clean_resume(text)
            #print(f"Longitud del currículum limpio: {len(clean_resume)}")

            clean_resume = limpiador.clean_useless_words(clean_resume, categoria)
            #print(f"Longitud del currículum limpio después de eliminar palabras irrelevantes: {len(clean_resume)}")

            clasificador = ModeloClasificador('svm_model.joblib', 'tfidf_vectorizer.joblib', 'selected_features.joblib', 'pca.joblib', 'label_encoder.joblib')
            predicted_category = clasificador.predecir(clean_resume)
            print(f"Categoría predicha del currículum: {predicted_category}")

        except FileNotFoundError:
            print(f"Error: Archivo PDF '{file_path}' no encontrado.")
        except Exception as e:
            print(f"Error inesperado: {str(e)}")

def main():
    path = "diccionarios"
    categoria = "Dotnet Developer"
    limpiador = LimpiadorCV(path)

    pdf_files = [
        'Pruebas_con_PDFs/Testing.pdf',
        'Pruebas_con_PDFs/Blockchain.pdf',
        # Agrega aquí más rutas de archivos según sea necesario
    ]

    clasificar_cv(pdf_files, categoria, limpiador)

if __name__ == "__main__":
    main()
