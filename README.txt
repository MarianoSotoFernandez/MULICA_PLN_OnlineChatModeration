

##########################################
 PROCESAMIENTO DEL LENGUAJE NATURAL (PLN)
##########################################

Autor: MARIANO SOTO FERNÁNDEZ
Fecha: Mayo 2026

##########################################

Tarea 2: Moderación automática de comunidades online
--------

Este código constituye un sistema automático de análisis de texto para la detección
de mensajes con carácter abusivo dentro de comunidades online.

##########################################

Dentro de este repositorio se proporcionan tanto los ficheros ".py" de cada implementación,
como otros programas para realizar el entrenamiento y comparación de los modelos y una
prueba sobre el model, tanto manual como automática.


##########################################
 ESTRUCTURA DEL DIRECTORIO
##########################################

Todos los contenidos del proyecto se almacenan de la siguiente forma:
	> Código: todo contenido en el directorio raíz ('.').
	> Datos: el conjunto de datos utilizado para entrenar y valorar los modelos
		 está contenido en "./data/".
	> Modelos: las configuraciones de los modelos producidos se almacenan dentro de 
		   "./models", concretamente "./models/tfidf_model" para "TF-IDF + SVM" y 
		   "./models/xlmroberta_models" para XLM-RoBERTa con fine-tunning.


##########################################
 DESCRIPCIÓN DE FICHEROS
##########################################

A continuación se describe la funcinalidad de cada fichero del proyecto:
	> "./data/train.csv": conjunto de datos de trabajo.
	> "./models/tfidf_models/*.joblib": model de TF-IDF entrenado.
	> "./models/xlm_roberta_models/*.pt": mejor configuración de pesos encontrada.
	> "./models/xlm_roberta_models/*tokenizer.json": definición del tokenizador.
	> "./modelss/xlm_roberta_models/tokenizer_config.json": configuración del tokenizador.
	> "abuse_classifier.py": wrapper para encapsular el clasificardor de tipo de abuso.
				 Permite utilizar cualquier de los dos modelos sin restrinciones.
	> "language_detector.py": programa encargado de realizar la idenficicación del lenguaje
				  de los mensajes
	> "main.py": archivos principal para la realización de pruebas del sistema. Posibilita
		     La ejecución automática de inputs ya preestablecidos, así como la posibilidad
		     de introducir datos por el usuario.
	> "README.txt": este propio documento.
	> "system_core.py": fichero que guarda la definición de la clase principal del sistema. Gestiona
			    la composición de los sistemas y estable el flujo de operatividad del sistema.
	> "TF_IDF_SVM.py": implementación del modelo TF-IDF + SVM.
	> "train_and_evaluate.py": Programa para realizar el entrenamiento de cada modelo, así como una breve
				   comparación de sus rendimientos.
	> "user_detector.py": programa dedicado a la detección de usuarios implicados en la comunicación
			      mediante un sistema NER.
	> "utils.py": fichero que contiene funciones varias esenciales para diversas partes del sistema (carga de dataset, visualización de datos, preprocesador, etc.).
	> "XLM_RoBERTa.py": implementación del modelo XLM-RoBERTa + fine-tunning.
	

##########################################
 DEPENDENCIAS
##########################################
 Para ejecutar el sistema son necesarias las siguientes dependencias de python:
	> os
	> lingua
	> time
	> joblib
	> numpy
	> sklearn
	> argparse
	> warnings
	> torch
	> spacy
	> re
	> pandas
	> transformers


##########################################
 MÉTODOS DE USO
##########################################

En general, el uso actual del sistema se reduce a ejecutar "train_and_evaluate.py" y "main.py", y seguir las
indicaciones que salgan por terminal.G

Entrenamiento:
--------------
 Ejecutar "train_and_evaluate.py". Puede recibir los argumentos:
	> "--sample": ajusta el tamaño del conjunto a utilizar.
	> "--skip-tfidf": permite evitar el entreno de TF-IDF.
	> "--skip-xlmr": permite evitar el entreno de XLM-RoBERTa.
	> "--train-path": permite ajustar el path al conjunto de datos.

Tiempo estimado (GPU de referencia NVIDIA 4060 laptop):
	> TF-IDF + SVM: 2~3 minutos.
	> XLM-RoBERTa-base + fine tuning: ~2 horas

Pruebas:
--------
Ejecutar "main.py" y seguir los pasos que aparecen por terminal.

Tiempo estimado (GPU de referencia NVIDIA 4060 laptop):
	> TF-IDF + SVM: < 0.03s.
	> XLM-RoBERTa-base + fine tuning: ~25s.


