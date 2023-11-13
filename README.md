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

### 01_preparacion_datos.ipynb

### 02_crea_ficheros_parametros.ipynb

### 02_explorar_datos.ipynb

### 03_entrenamiento.ipynb

### 04_1_sistema_1fases.ipynb

### 04_2_sistema_2fases.ipynb

### 04_3_sistema_2fases_multimodelo.ipynb

### 06_graficas_modelos_semillas.ipynb

### 06_graficas_modelos_semilla_0042.ipynb

## Orden de ejecución

El número en el nombre de los notebooks indica el orden de ejecución.

## Ejecución del entrenamiento desde la línea de comando

Para la ejecución desde la línea de comando se usa la utilidad "papermill".

Hay dos scripts para poder lanzar la ejecución del notebook de entrenamiento pasandole un fichero con los parámetros a utilizar.

## Creación de los ficheros de parámetros para la ejecucion con papermill

Se incluye el notebook "02_crea_fichero_parametros" (a modo de ejemplo) que lee los parámetros de un fichero CSV para inyectar al notebook que hace el entrenamiento cuando se llame con papermill.

## Lanzar el entrenamiento desde script

Hay creados dos scripts para la ejecución desde un entorno windows o linux:

* Windows: **dos_ejecuta_modelos.cmd**

* Linux: **ejecuta_modelos.sh**

Ambos scripts se lanzan desde la línea de comando sin necesidad de parámetros. Utilizará por defecto la carpeta "ejecuciones".
