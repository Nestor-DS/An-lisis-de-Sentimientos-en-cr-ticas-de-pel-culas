# Anlisis-de-Sentimientos-en-crticas-de-peliculas
# 1.	Dataset (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):
El dataset a ocupar es “IMDB Dataset of 50K Movie Reviews”, este es un conjunto de datos IMDB con 50.000 críticas de películas para el procesamiento del lenguaje natural o el análisis de textos. 
Se trata de un conjunto de datos para la clasificación binaria de sentimientos que contiene muchos más datos que los anteriores conjuntos de datos de referencia, consta de dos columnas de datos, la primera de “review” y la segunda de “sentiment”. Proporciona un conjunto de 25.000 críticas de películas altamente polares para el entrenamiento y 25.000 para las pruebas. (Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts, 2011)
Este dataset es adecuado para predecir el número de críticas positivas y negativas utilizando algoritmos de clasificación o de aprendizaje profundo.

# 2.	Problemas para analizar:

# <h2> 2.1 Contar el número de tipos de palabras utilizados en las críticas (adjetivos, verbos, conectores, etc.): </h2>
El objetivo es analizar el contenido lingüístico de las críticas de películas y determinar cuántos adjetivos, verbos, conectores y otros tipos de palabras se utilizan en el conjunto de datos. Esto proporcionará información sobre la variedad de vocabulario y estructuras lingüísticas presentes en las críticas.
2.2 Encontrar las palabras más usadas en las reviews:
Este análisis busca identificar las palabras más frecuentes en todas las reseñas de películas. Al hacerlo, se podrán obtener ideas sobre las palabras clave o temas comunes que aparecen con mayor frecuencia en las opiniones de las películas.

# <h2> 2.3 Distribución de frecuencias de palabras en una reseña: </h2>
El objetivo de esta tarea es examinar cómo se distribuyen las frecuencias de las palabras en una reseña específica. Esto puede proporcionar información sobre qué palabras son más repetitivas o destacadas en una reseña particular.

# <h2> 2.4	Contar la cantidad de reseñas positivas y negativas: </h2>
El objetivo de esto es buscar y determinar cuántas reseñas de películas son consideradas positivas y cuántas son negativas. Esto permitirá entender la proporción de opiniones positivas y negativas en el conjunto de datos y tener una idea general sobre la polaridad de las reseñas.

# <h2> 2.5	Encontrar los adjetivos más usados en las reseñas:</h2>
El objetivo es identificar los adjetivos más comunes utilizados en las reseñas de películas. Al hacerlo, se podrá comprender qué tipo de descripciones o calificativos se destacan con mayor frecuencia en las opiniones.

# <h2> 2.6	Obtener todos los adjetivos de las reseñas positivas y negativas:</h2>
El objetivo de esto es busca identificar los adjetivos más utilizados en las reseñas positivas y negativas por separado. Comparar los adjetivos en cada tipo de reseña puede ayudar a entender qué aspectos específicos son más apreciados o criticados por los espectadores.

# <h2> 2.7	Calcular la longitud de las reseñas (número de palabras):</h2>
El objetivo de esto es el análisis de la cantidad de palabras en cada reseña. Esto permitirá obtener una perspectiva de la extensión promedio de las opiniones, lo que puede ser útil para comprender cuánto detalle se proporciona en las reseñas.

# 4	Carga de librerías:
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/6daf9a12-ae56-4226-9a68-c3f631783701)

# 5	Carga y limpieza de datos:

Primero, utilizamos la biblioteca panda para cargar el dataset y almacenarlo en la variable df. 
Una vez que tenemos el DataFrame cargado, procedemos a limpiar los datos. En este caso, identificamos los saltos de línea en las columnas de texto y los reemplazamos por espacios en blanco. Esto es importante para mejorar la consistencia del texto y facilitar su procesamiento posterior.
 ![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/1cdde419-6a28-4476-ac3c-e5adf42b6209)

# 6	División de texto en oraciones
Ahora, se presenta la implementación de la tarea de división de texto en oraciones utilizando la biblioteca spaCy. El objetivo de esta tarea es tomar las reseñas limpias almacenadas en el DataFrame y dividirlas en oraciones para un análisis de lenguaje más detallado.
Se inicia la tarea importando la biblioteca spaCy y cargando el modelo 'en_core_web_sm'. Se define una función llamada spacy_sentence_tokenize(text). Esta función toma un texto como entrada y utiliza el modelo cargado de spaCy para dividir el texto en oraciones. Cada oración resultante se almacena en una lista.
Una vez que se ha definido la función de tokenización de oraciones, se aplica a la columna 'cleaned_review' del DataFrame df, que contiene las reseñas de películas limpias. La función procesará cada reseña y devolverá una lista de oraciones correspondientes a cada reseña.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/4a664093-f938-4cc5-814a-181a885d3c8f)

# 7	Etiquetación de un texto en palabras
Definiremos una una función llamada spacy_pos_tagging(text), que toma un texto como entrada y utiliza el modelo de spaCy previamente cargado para etiquetar cada palabra con su parte del discurso (POS).
Por medio de La función spacy_pos_tagging se aplica a la columna 'cleaned_review' del DataFrame df, que contiene las reseñas de películas limpias. La función procesa cada reseña y etiqueta cada palabra con su etiqueta POS correspondiente, generando una lista de tuplas para cada reseña.
A continuación, se crea un nuevo DataFrame llamado tagged_words_df, donde se almacenan todas las palabras etiquetadas con sus respectivas etiquetas POS. Se utiliza la función explode() para convertir la lista de tuplas en columnas individuales, y luego se guardan las palabras y etiquetas en columnas separadas llamadas 'palabra' y 'etiqueta'.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/e1c34560-b8a4-4558-800f-d946c1f12477)

# 8	Número de tipos de palabras utilizados en las críticas (adjetivos, verbos, conectores, etc.)
Se carga el modelo 'en_core_web_sm' de spaCy y se definió la función spacy_pos_tagging(text) para etiquetar cada palabra en el texto con su parte del discurso (POS).
Se crea un nuevo DataFrame llamado tagged_words_df para almacenar las palabras etiquetadas y sus etiquetas POS. Se utilizó la función explode() para convertir la lista de tuplas en columnas individuales.
Se clasifican las palabras etiquetadas por tipo (adjetivo, número, etc.) y se contó la cantidad de palabras de cada tipo utilizando la función value_counts().
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/a70a085a-f59d-45a1-8dea-7c4a90fdff1e)

#<h2>8.1	Cantidad de palabras (Gráfica)</h2>
La gráfica de barras nos permite tener una visión general de las categorías gramaticales presentes en las reseñas con el fin de entender cómo las personas que realizaron las reseñas utilizan diferentes tipos de palabras para expresar sus opiniones sobre las películas. 
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/a659d288-6786-4a79-988d-52f541e4a3ee)

# 9	Palabras más empleadas en las reseñas
Se crea una lista llamada all_words para almacenar todas las palabras presentes en las reseñas.
Mediante la función FreqDist de NLTK se calcula la frecuencia de cada palabra en la lista all_words. FreqDist y se crea un diccionario que asigna cada palabra a su frecuencia en el texto.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/28e97bdb-66d1-4764-8961-595cd9eb0584)

# <h2>9.1 Frecuenta de palabras (Gráfica)</h2>
Gracias a la siguiente grafica podemos observar las palabras más frecuentes en las reviews, lo que puede ser útil para entender los patrones de lenguaje y los temas comunes que las personas mencionan con mayor frecuencia.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/b3daaba6-97cf-4361-bfee-bdd05572c967)

# 10	Resumen del texto
El resumen se refiere a un breve extracto o versión abreviada del contenido de las reseñas de películas.
Se crea una función auxiliar llamada get_luhn_summary(text, sentences_count=2), que toma un texto como entrada y utiliza el algoritmo LuhnSummarizer para generar un resumen del texto. El parámetro sentences_count indica la cantidad de oraciones que se desean en el resumen, y por defecto se estableció en 2.
Mediante la función get_luhn_summary se aplica a la columna 'cleaned_review' del DataFrame df, que contiene las reseñas de películas limpias con el fin de crear un resumen para cada reseña y almacenarlo en una nueva columna llamada 'text_summary'.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/952f867a-b005-46dc-8b9f-734e0e60083f)

# 11	Distribución de frecuencias de palabras
Mediante la función word_tokenize de NLTK se tokeniza cada reseña limpiada en el DataFrame df. Luego, se aplica la función FreqDist a la lista de tokens resultante para calcular la frecuencia de cada palabra.
Esto nos permite tener una visión de las distribuciones de frecuencias de palabras en las reseñas de películas. Esta información puede ser útil para comprender qué palabras se utilizan con mayor frecuencia en las reseñas y cómo se distribuyen las palabras en cada una de ellas.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/1098cfb8-75e5-4ab5-a162-03db0ce75d19)

# 12	Cantidad de "positive" y "negative"
Mediante la función value_counts() de pandas contamos la cantidad de ocurrencias de cada etiqueta en la columna 'sentiment' con el fin de obtener la cantidad reseñas positivas y negativas.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/bb4552fa-223a-4b5a-800a-80a17964a1d1)

# <h2>12.1 Cantidad de positivos y negativos (Gráfica)</h2>
Mediante la siguiente grafica podemos observar la cantidad de reseñas positivas y negativas, esto puede ser útil para entender la reacción general del público hacia las películas y realizar análisis de sentimiento o evaluaciones de opiniones.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/02bc0b76-e046-4ae2-95eb-c65885014308)

# 13	Número de palabras en cada reseña
Ahora se realiza el cálculo del número de palabras en cada reseña de películas y se creó una gráfica de dispersión para visualizar la distribución del número de palabras en todas las reseñas.
Mediante la función apply() de pandas junto con la función len(x.split()) calculamos el número de palabras en cada reseña y con la función scatter() de matplotlib se crear una gráfica de dispersión.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/77e7d586-88a5-4f60-8845-9e551ca98b48)

# <h2>13.1 Número de palabras en reseñas</h2>
La siguiente gráfica de dispersión proporciona una visión de la distribución del número de palabras en las reseñas de películas, esto puede ser útil para entender la longitud promedio de las reseñas y la variabilidad en las longitudes.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/1988fe22-839c-447b-a6f3-a2cdbc00b6df)

# 14	Adjetivos más Repetidos en Reseñas Positivas
Ahora se crea un análisis de los adjetivos presentes en las reseñas positivas del conjunto de datos. 
Para esto primero seleccionamos las reseñas del DataFrame df que tienen la etiqueta 'positive' en la columna 'sentiment', y se almacenaron en la variable positive_reviews. 
Por medio de la función get_adjectives(text) para obtenemos los adjetivos de una reseña. Primero, se tokeniza el texto en palabras individuales utilizando word_tokenize() y se filtran solo las palabras con caracteres alfabéticos utilizando una expresión regular.
Se crea una función FreqDist para calcular la frecuencia de cada adjetivo en la lista positive_adjectives, creando así un objeto FreqDist que almacena la frecuencia de cada adjetivo.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/def4fdd8-cdfa-43ef-8c01-142380be7d47)

# <h2>14.1 Adjetivos más repetidos en reseñas positivas (Gráfica)</h2>
La siguiente gráfica de barras nos permite identificar los adjetivos más comunes en las reseñas positivas y proporciona información relevante sobre cómo las personas expresan sus opiniones positivas acerca de las películas.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/19b0b4c4-59e8-47f9-98d9-d7259bd1c5e3)

# 15	Adjetivos más Repetidos en Reseñas Negativas
Ahora realizamos un análisis similar al realizado para las reseñas positivas, pero enfocándonos en las reseñas negativas. Identificaron los 10 adjetivos más repetidos en las reseñas con sentimiento negativos.
Mediante el objeto negative_adj_freq se asumimos que ya fue calculado previamente, para obtener los 10 adjetivos más frecuentes en las reseñas negativas utilizando most_common(10).
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/a2812439-99a4-4a51-9f64-8c153849f885)

#<h2> 15.1 Adjetivos más repetidos en reseñas negativas (Gráficas)</h2>
La siguiente gráfica de barras nos permite identificar los adjetivos más comunes en las reseñas negativas proporcionando información relevante sobre cómo las personas expresan sus opiniones negativas acerca de las películas.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/7c1981da-4dee-403f-a9a6-ef90ff3e0404)

# 16	Longitud de las reseñas
Por último, mediante un diagrama de cajas creamos un análisis comparativo de la longitud de las reseñas positivas y negativas en función del número de palabras que contienen con el fin de visualizar la distribución de las longitudes de las reseñas en ambos grupos de sentimientos.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/54936e73-82ec-41bd-ba3c-424f631077f7)

# <h2>16.1	Distribución de longitud (Diagrama de cajas)</h2>
De acuerdo al diagrama presentado podemos asumir que las reseñas positivas tienden a tener una longitud de palabras ligeramente mayor en comparación con las reseñas negativas, ya que la mediana de las reseñas positivas se encuentra en una posición ligeramente más alta que la mediana de las reseñas negativas.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/5830e5ee-6ab7-4042-a128-f15d499fb128)


