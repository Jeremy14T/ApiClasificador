import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, RFE
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class ModeloClasificador:
    def __init__(self, modelo_path, vectorizer_path, selector_path, svd_path, label_encoder_path):
        # Cargar los componentes del pipeline
        self.vectorizer = joblib.load(vectorizer_path)
        self.selector = joblib.load(selector_path)
        self.svd = joblib.load(svd_path)
        self.modelo = joblib.load(modelo_path)
        self.label_encoder = joblib.load(label_encoder_path)

    
    def predecir(self, text):
        # Aplicar el vectorizador Tfidf a los datos
        text_vectorized = self.vectorizer.transform([text])
        
        # Seleccionar las características según la Información Mutua
        text_selected = text_vectorized[:, self.selector]
        
        # Reducir la dimensionalidad con PCA/SVD
        text_reduced = self.svd.transform(text_selected)
        
        # Realizar la predicción con el modelo
        prediction_numeric = self.modelo.predict(text_reduced)
        
        # Convertir la predicción numérica a la etiqueta original
        prediction_label = self.label_encoder.inverse_transform(prediction_numeric)
        
        return prediction_label


if __name__ == "__main__":
    #Cargar dataset
    df = pd.read_csv('Dataset_Final.csv', encoding="utf-8")

    # Separar las columnas de características y etiquetas
    X = df[['Clean_Resume', 'Common Skills']]

    df1 =df.copy()
    le = LabelEncoder()

    le.fit(df['Category'])
    df1['Category_num'] = le.transform(df['Category'])
    df['Category'] = le.transform(df['Category'])

    y= df['Category']


    # Obtener el tipo de datos de una columna específica
    #print(df['Clean_Resume'].dtype)

    # Convertir las columnas de características a texto
    X['Clean_Resume'] = X['Clean_Resume'].astype(str)
    X['Common Skills'] = X['Common Skills'].astype(str)

    # Obtener el tipo de datos de una columna específica
    #print(df['Clean_Resume'].dtype)

    # Aplicar TfidfVectorizer a las columnas de características
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X['Clean_Resume'] + ' ' + X['Common Skills'])

    # Divide el dataset en train y test
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25)

    # Convertir datos dispersos a matriz densa
    X_train_dense = X_train.toarray()

    mi_scores = []

    # Calcular la MI para cada característica
    for feature in X_train_dense.T:
        mi_scores.append(mutual_info_score(feature, y_train))

    mi_scores = np.array(mi_scores)
    top_features_mi = np.argsort(mi_scores)[-50:]

    #print(f"Top features mutual information: {top_features_mi}")

    # Aplicar esta selección a tus conjuntos de datos
    X_train_selected = X_train[:, top_features_mi]
    X_test_selected = X_test[:, top_features_mi]


    svd = TruncatedSVD(n_components=16, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_selected)


    # Luego puedes transformar tu conjunto de prueba con el mismo ajuste
    X_test_reduced = svd.transform(X_test_selected)


    # Calcular la varianza explicada acumulativa
    varianza_explicada_cumulativa = np.cumsum(svd.explained_variance_ratio_)

    # Graficar la varianza explicada acumulativa
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(varianza_explicada_cumulativa)+1), varianza_explicada_cumulativa, marker='o', linestyle='-')
    plt.title('Análisis del Codo para PCA')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada Acumulativa')
    plt.grid(True)
    plt.show()

    # Instanciar el modelo RandomForest
    rf_model = RandomForestClassifier(random_state=42)

    # Entrenar el modelo en los datos de entrenamiento reducidos
    rf_model.fit(X_train_reduced, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = rf_model.predict(X_test_reduced)

    # Decodificar las etiquetas numéricas a texto en tus predicciones
    y_pred_labels = le.inverse_transform(y_pred)

    # Decodificar también las etiquetas verdaderas si es necesario
    y_test_labels = le.inverse_transform(y_test)

    # Calcular el reporte de clasificación usando las etiquetas de texto
    reporte = classification_report(y_test_labels, y_pred_labels, output_dict=True)

    # Imprimir el reporte de clasificación con etiquetas de texto
    print(classification_report(y_test_labels, y_pred_labels))

    # Opcional: Imprimir la matriz de confusión
    print(confusion_matrix(y_test_labels, y_pred_labels))

    # No es necesario definir una cuadrícula de hiperparámetros específicos para SVM, ajustaremos para RandomForest si necesario
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }

    # Instanciar el Grid Search
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=2, cv=5)

    # Ejecutar el grid search
    grid_search.fit(X_train_reduced, y_train)

    # Guardar los componentes necesarios del preprocesamiento y el modelo
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(top_features_mi, 'selected_features.joblib')
    joblib.dump(svd, 'pca.joblib')
    joblib.dump(rf_model, 'rf_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')        

