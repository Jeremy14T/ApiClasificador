import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import WhitespaceTokenizer

import plotly.graph_objects as go
import plotly.express as px
import os

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings('ignore')


class LimpiadorCV:
    def __init__(self, carpeta_diccionarios):
        self.diccionarios = self.cargar_diccionarios(carpeta_diccionarios)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def cargar_diccionarios(self, carpeta_diccionarios):
        diccionarios = {}
        for archivo in os.listdir(carpeta_diccionarios):
            categoria = archivo.split('.')[0]
            with open(os.path.join(carpeta_diccionarios, archivo), 'r', encoding='utf-8') as f:
                diccionarios[categoria.lower()] = set(palabra.strip().rstrip(',') for palabra in f.read().splitlines())
        return diccionarios
                
    def clean_resume(self, resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # remove non-ascii characters
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        resumeText = re.sub(r'\b\d+\b', ' ', resumeText)  # remove numbers
        resumeText = resumeText.lower() # convert to lowercase

        word_tokens = word_tokenize(resumeText)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_text]
        
        return ' '.join(lemmatized_tokens)

    def clean_useless_words(self, resumen, categoria):
        palabras_irrelevantes = self.diccionarios.get(categoria.lower(), set())
        palabras_limpias = [palabra for palabra in resumen.split() if palabra not in palabras_irrelevantes]
        return ' '.join(palabras_limpias)


if __name__ == '__main__':
    
    # Cargar el DataFrame de tu archivo CSV
    df = pd.read_csv('path_to_youhhr_modified.csv', encoding="utf-8")

    # Uso de la clase LimpiadorCV
    carpeta_diccionarios = 'diccionarios'
    limpiador_cv = LimpiadorCV(carpeta_diccionarios)

    # Asumiendo que df es tu DataFrame después de las operaciones previas
    df['Clean_Resume'] = df['Clean_Resume'].apply(limpiador_cv.clean_resume)
    df['Clean_Resume'] = df.apply(lambda fila: limpiador_cv.clean_useless_words(fila['Clean_Resume'], fila['Category']), axis=1)



    '''
    print(data.shape)
    data.describe()
    data.info()

    #Revisar nulos
    data[data.isna().any(axis=1) | data.isnull().any(axis=1)]

    data.nunique()
    valores_unicos = data.Category.unique()
    '''

    data = df[df['Category'].apply(lambda x: len(x.split()) <= 5)]

    valores_a_eliminar = [
        'Automation Testing ',
        'Designer ',
        'ETL Developer ',
        'Information Technology ',
        'Project manager ',
        'SAP Developer ',
        'Security Analyst ',
        'Web Developer '
    ]

    '''valores_a_eliminar = [
        'HR',
        'Advocate',
        'Arts',
        'Business Analyst',
        'SAP Developer',
        'Automation Testing',
        'Electrical Engineering',
        'Operations Manager',
        'PMO',
        'Hadoop',
        'ETL Developer',
        'Mechanical Engineer',
        'Sales',
        'Health and fitness',
        'Civil Engineer',
        'ETL Developer'
    ]'''

    df = data[~data['Category'].isin(valores_a_eliminar)]

    '''
    #imprimimos las etiquetas nuevas que quedaron
    df['Category'] = df['Category'].str.rstrip()

    valores_unicos = df.Category.unique()
    '''

    '''
    df['Clean_Resume'] = df["Clean_Resume"].map(cleanResume)

    df['Clean_Resume'] = df['Clean_Resume'].str.replace('Â', '').str.replace('â', '')

    # Ruta a la carpeta que contiene tus archivos de diccionarios
    carpeta_diccionarios = 'diccionarios'
    '''

    # Cargar los diccionarios de palabras irrelevantes desde los archivos
    diccionarios = {}

    '''
    for archivo in os.listdir(carpeta_diccionarios):
        categoria = archivo.split('.')[0]  # Asume que el nombre del archivo es exactamente la categoría
        with open(os.path.join(carpeta_diccionarios, archivo), 'r', encoding='utf-8') as f:
            # Crea un conjunto de palabras irrelevantes para cada categoría
            # Elimina las comas de cada palabra antes de agregarlas al conjunto
            diccionarios[categoria] = set(palabra.strip().rstrip(',') for palabra in f.read().splitlines())

            
    #NUBE ANTES DE APLICAR EL DICCIONARIO
    # Filtrar los resúmenes por la etiqueta 'X'
    all_resumes = df[df['Category'] == 'Python Developer']['Clean_Resume']

    # Combinar todos los resúmenes en una única cadena de texto
    combined_resumes = ' '.join(all_resumes)

    # Crear y visualizar la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_resumes)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # No mostrar los ejes para una visualización más limpia
    plt.show()

    #APLICAR DICCIONARIOS
    # Aplicar la función de limpieza a cada fila del DataFrame
    df['Clean_Resume'] = df.apply(lambda fila: clean_ussles_words(fila['Clean_Resume'], fila['Category']), axis=1)# revisar si se aplicaron lso diccionarios

    #NUBE DESPUES DE APLICAR EL DICCIONARIO
    # Filtrar los resúmenes por la etiqueta 'X'
    all_resumes = df[df['Category'] == 'Python Developer']['Clean_Resume']

    # Combinar todos los resúmenes en una única cadena de texto
    combined_resumes = ' '.join(all_resumes)

    # Crear y visualizar la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_resumes)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # No mostrar los ejes para una visualización más limpia
    plt.show()

    skills_dict = {
        'Blockchain': 'Ethereum Smart Contracts, Cryptography, Solidity, DApp Development, Hyperledger Fabric, Consensus Algorithms, Node.js, Corda, Blockchain System Design, Truffle, Ganache, Blockchain API, Ripple, Litecoin, Bitcoin, Blockchain Security, Key Management, DeFi, NFTs',
        'Data Science': 'Python, R, Machine Learning, Deep Learning, Statistical Analysis, pandas, NumPy, Data Visualization, Matplotlib, Seaborn, SQL, NoSQL, Spark, Hadoop, PySpark, Ensemble Models, PCA, t-SNE, Airflow, Keras, TensorFlow, Time Series, Predictive Analysis, A/B Testing',
        'Database': 'SQL, PL/SQL, Database Administration, MySQL, PostgreSQL, Oracle, MongoDB, Cassandra, Database Design, Normalization, Data Recovery, Query Optimization, Distributed Databases, Database Tuning, Replication, Sharding, Cluster Management, ETL, Data Warehousing, Real-time DB, Redis, Database Security, Compliance',
        'DevOps Engineer': 'CI/CD, Jenkins, GitLab CI, Docker, Kubernetes, Ansible, Terraform, Monitoring, Logging, Prometheus, Grafana, ELK, Unix Administration, Bash, Python, CloudFormation, Spinnaker, Secrets Management, Ruby, Perl, Vagrant',
        'DotNet Developer': 'C#, .NET Framework, ASP.NET MVC, Entity Framework, LINQ, Azure, WPF, Windows Forms, RESTful APIs, .NET Core, Microservices, SignalR, RabbitMQ, CI/CD, TeamCity, Xamarin, Stress Testing',
        'Java Developer': 'Java, Spring Framework, Spring Boot, Hibernate, JPA, Maven, Gradle, SOAP, REST, JUnit, Design Patterns, Jersey, Netflix OSS, Spring Security, Android Development, SQL, NoSQL Integration, JVM Optimization',
        'Network Security Engineer': 'Firewalls, VPNs, IDS/IPS, SSL/TLS, SSH, Vulnerability Assessment, Pentesting, Intrusion Detection, ISO 27001, NIST, Cryptography, SIEMs, Digital Forensics, Firewall Configuration, Network Security Policies, DLP, CISSP, CISM',
        'Python Developer': 'Python, Django, Flask, Scripting, Data Analysis, Pandas, NumPy, Web Scraping, BeautifulSoup, Scrapy, API Development, FastAPI, Django REST, pytest, unittest, Pyramid, Bottle, asyncio, ORM, SQLAlchemy, Packaging, Kafka, RabbitMQ',
        'Testing': 'HTML, CSS, JavaScript, UX, UI, Frameworks, Graphic Design, Photoshop, Illustrator, Logo Design, Prototyping, Accessibility, SEO, Management, Test Animation, Optimization, Security, Responsive Design',
        'Web Designing': 'HTML, CSS, JavaScript, Responsive Design, UX, UI, Bootstrap, Foundation, Graphic Design, Photoshop, Illustrator, Logo Design, Prototyping, Accessibility, SEO, WordPress, Git, Web Animation, Debugging, Performance Optimization, Web Security'
    }

    # Agregar la columna de habilidades comunes
    df['Common Skills'] = df['Category'].apply(find_common_skills)

    #hacer todo minusculas
    df["Common Skills"] = [
        [palabra.lower() for palabra in lista] for lista in df["Common Skills"]
    ]


    df["Common Skills"] = df["Common Skills"].str.join(" ")

    print(df["Common Skills"])

    # Mantén solo las columnas que deseas, eliminando el resto
    df = df[['Category', 'Clean_Resume', 'Common Skills']]
    '''

    #df.to_csv("fasdfaesfasef.csv", index=False) #Cambiar a pp o jeremy depediendo el datset de entrada
    #analizar el otro datset en las nubees


