# Trabajo fin de máster

## Creación y evaluación de modelos para la clasificación de diferentes cánceres a partir de biopsias líquidas

## Paquetes necesarios

Para la ejecución de los notebook se ha instalado el paquete "miniconda".

A la instalación estándar se han añadido las siguientes librerías:

* numpy
* pandas
* scikit-learn
* scikit-learn-intelex
* seaborn
* matplotlib
* jupyter
* pytables
* tensorflow
* nbconver
* papermill
* imbalanced-learn

## Listado de notebooks

### 00_descarga_datos.ipynb

Descarga de intenet los ficheros con los datos.

### 01_preparacion_datos.ipynb

Proceso de preparación de los datos para el entrenamiento de los modelos

### 02_crea_ficheros_parametros.ipynb

Generación de los ficheros de parámetros para usar con papemill
Se puede encontrar un fichero de ejemplo con parámetros en ".\ejecuciones\parametros\02_parametros_definitivos.csv"

### 02_explorar_datos.ipynb

Exploración de los datos de los ficheros.

### 03_entrenamiento.ipynb

Notebook para realizar el entrenamiento de los modelos.

### 04_1_sistema_1fases.ipynb

Sistema de clasificación con un único modelo

### 04_2_sistema_2fases.ipynb

Sistema de clasificación en 2 fases, con un modelo por fase

### 04_3_sistema_2fases_multimodelo.ipynb

Sistema de clasificación en 2 fases multimodelo

### 06_graficas_modelos_semillas.ipynb

Revisión de los datos de métricas para los modelos con las diferentes semillas.

### 06_graficas_modelos_semilla_0042.ipynb

Revisión de los datos de métricas para los modelos con la semilla 42.

## Orden de ejecución

El número en el nombre de los notebooks indica el orden de ejecución.

## Creación de los ficheros de parámetros para la ejecucion con papermill

Se incluye el notebook "02_crea_fichero_parametros" (a modo de ejemplo) que lee los parámetros de un fichero CSV para inyectar al notebook que hace el entrenamiento cuando se llame con papermill.

## Ejecución del entrenamiento desde la línea de comando

Para la ejecución desde la línea de comando se usa la utilidad "papermill".

Hay dos scripts para poder lanzar la ejecución del notebook de entrenamiento pasandole un fichero con los parámetros a utilizar.

## Lanzar el entrenamiento desde script

Hay creados dos scripts para la ejecución desde un entorno windows o linux:

* Windows: **dos_ejecuta_modelos.cmd**

* Linux: **ejecuta_modelos.sh**

Ambos scripts se lanzan desde la línea de comando sin necesidad de parámetros. Utilizará por defecto la carpeta "ejecuciones".
