import os
import platform
import sys
import numpy as np
import pandas as pd
import datetime
import psutil
import sklearn
import logging
import PIL
import seaborn as sns
import matplotlib
import IPython
import joblib

from matplotlib import pyplot as plt
from IPython import get_ipython
from IPython.display import display

print(f"CPU_COUNT: {os.cpu_count()}; NODE: {platform.node()}; sys.version: {sys.version}")

LOKY_MAX_CPU_COUNT = os.environ.get("LOKY_MAX_CPU_COUNT")
if LOKY_MAX_CPU_COUNT:
    print("LOKY_MAX_CPU_COUNT:", LOKY_MAX_CPU_COUNT)
else:
    print("LOKY_MAX_CPU_COUNT is not set.")

OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS")
if OMP_NUM_THREADS:
    print("OMP_NUM_THREADS:", OMP_NUM_THREADS)
else:
    print("OMP_NUM_THREADS is not set.")

NOTEBK_FILENAME = "myfunctions.py"
N_ROWS = None
START_CHKPOINT = datetime.datetime.now()
LAST_CHKPOINT = datetime.datetime.now()
END_CHKPOINT = datetime.datetime.now()

HOME_DIR     = os.path.join("..","tfm_viu")

DATA_DIR     = os.path.join(HOME_DIR,"datos")
CFDNA_DIR    = os.path.join(DATA_DIR,"cfDNA_5hmC")
GENCODE_DIR  = os.path.join(DATA_DIR,"gencode")
H5_DIR       = os.path.join(DATA_DIR,"h5")
LOG_DIR      = os.path.join(DATA_DIR,"logs")
CSV_DIR      = os.path.join(DATA_DIR,"csv")

EXEC_DIR     = os.path.join(HOME_DIR,"ejecuciones")
MET_DIR      = os.path.join(EXEC_DIR,"metricas")
MODEL_DIR    = os.path.join(EXEC_DIR,"modelos")

LOGGING = 0
SHOW_DISPLAYS = 0
TARGETS = ["Control","Bladder","Breast","Colorectal","Kidney","Lung","Prostate"]
TARGET_MAPPING = {'Control': 0, 'Bladder': 1, 'Breast': 2, 'Colorectal': 3, 'Kidney': 4, 'Lung': 5, 'Prostate': 6}
EXAMPLES_MAPPING = {'bin_s': 'Algunas muestras reales', 'bin_m': 'Algunas muestras reales y sintéticas', 
                    'mul_s': 'Algunas muestras reales ', 'mul_m': 'Algunas muestras reales y sintéticas'}
DATE_FORMAT='%Y%m%dT%H%M%S'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILENAME = os.path.join(LOG_DIR,str(N_ROWS)+"-"+datetime.datetime.now().strftime(DATE_FORMAT)+"-"+NOTEBK_FILENAME+'.log')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Obtener el nombre del playbook si se ejecutan en vscode 
# o extraerlo de la variable de entorno IPYNB_FILE
def get_nb_name():
    ip = get_ipython()
    path = None
    if '__vsc_ipynb_file__' in ip.user_ns:
        path = ip.user_ns['__vsc_ipynb_file__']
    if path is None:
        fichero1=os.environ['IPYNB_FILE']
    else:
        fichero1=os.path.basename(path)
    return fichero1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# mostrar el contenido de un DataFrame si está activa la variable SHOW_DISPLAYS
def show_display(df1) -> None:
    if SHOW_DISPLAYS:
        display(df1)
    else:
        print(f'Display desactivado. Pero te muestro el shape {df1.shape}')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# mostrar mensaje con posibilidad de llevar a un fichero de log
def verbose(text1: str) -> None:
    print(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'), ":", platform.node(), ":", "INFO", ":", text1)
    if LOGGING:
        logging.info(text1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Establecer el valor inicial de las variables
def reset_vars():
    global DATA_DIR
    global CFDNA_DIR
    global GENCODE_DIR
    global LOG_DIR
    global CSV_DIR
    global MODEL_DIR
    global EXEC_DIR
    global MET_DIR
    global LOG_FILENAME
    HOME_DIR     = os.path.join("..","tfm_viu")
    
    DATA_DIR     = os.path.join(HOME_DIR,"datos")
    CFDNA_DIR    = os.path.join(DATA_DIR,"cfDNA_5hmC")
    GENCODE_DIR  = os.path.join(DATA_DIR,"gencode")
    H5_DIR       = os.path.join(DATA_DIR,"h5")
    LOG_DIR      = os.path.join(DATA_DIR,"logs")
    CSV_DIR      = os.path.join(DATA_DIR,"csv")
    
    EXEC_DIR     = os.path.join(HOME_DIR,"ejecuciones")
    MET_DIR      = os.path.join(EXEC_DIR,"metricas")
    MODEL_DIR    = os.path.join(EXEC_DIR,"modelos")
    
    LOG_FILENAME = os.path.join(LOG_DIR,str(N_ROWS)+"-"+datetime.datetime.now().strftime(DATE_FORMAT)+"-"+NOTEBK_FILENAME+'.log')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Visualizar el contenido de las variables
def print_vars(name1=None):
    print("\n")
    if name1 == "NOTEBK_FILENAME" or name1 is None: print(f"NOTEBK_FILENAME: {NOTEBK_FILENAME}")
    if name1 == "START_CHKPOINT" or name1 is None:  print(f"START_CHKPOINT: {START_CHKPOINT}")
    if name1 == "N_ROWS" or name1 is None:          print(f"N_ROWS: {N_ROWS}")
    if name1 == "DATA_DIR" or name1 is None:        print(f"DATA_DIR: {DATA_DIR}")
    if name1 == "CFDNA_DIR" or name1 is None:       print(f"CFDNA_DIR: {CFDNA_DIR}")
    if name1 == "GENCODE_DIR" or name1 is None:     print(f"GENCODE_DIR: {GENCODE_DIR}")
    if name1 == "H5_DIR" or name1 is None:          print(f"H5_DIR: {H5_DIR}")
    if name1 == "LOG_DIR" or name1 is None:         print(f"LOG_DIR: {LOG_DIR}")
    if name1 == "CSV_DIR" or name1 is None:         print(f"CSV_DIR: {CSV_DIR}")
    if name1 == "MODEL_DIR" or name1 is None:       print(f"MODEL_DIR: {MODEL_DIR}")
    if name1 == "EXEC_DIR" or name1 is None:        print(f"EXEC_DIR: {EXEC_DIR}")
    if name1 == "MET_DIR" or name1 is None:         print(f"MET_DIR: {MET_DIR}")
    if name1 == "LOGGING" or name1 is None:         print(f"LOGGING: {LOGGING}")
    if name1 == "SHOW_DISPLAYS" or name1 is None:   print(f"SHOW_DISPLAYS: {SHOW_DISPLAYS}")
    if name1 == "TARGETS" or name1 is None:         print(f"TARGETS: {TARGETS}")
    if name1 == "TARGET_MAPPING" or name1 is None:  print(f"TARGET_MAPPING: {TARGET_MAPPING}")
    if name1 == "DATE_FORMAT" or name1 is None:     print(f"DATE_FORMAT: {DATE_FORMAT}")
    if name1 == "LOG_FORMAT" or name1 is None:      print(f"LOG_FORMAT: {LOG_FORMAT}")
    if name1 == "LOG_FILENAME" or name1 is None:    print(f"LOG_FILENAME: {LOG_FILENAME}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Comprobar que existen las carpetas necesarias
def check_enviroment(DATA_DIR, CFDNA_DIR, GENCODE_DIR, H5_DIR, LOG_DIR, CSV_DIR, MODEL_DIR, EXEC_DIR, MET_DIR):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
        verbose(f"Creada carpeta DATA_DIR={DATA_DIR}")
    else:
        verbose(f"Encontrada carpeta DATA_DIR={DATA_DIR}")

    if not os.path.exists(CFDNA_DIR):
        os.mkdir(CFDNA_DIR)
        # raise ValueError("No encontrada ruta",CFDNA_DIR)
    else:
        verbose(f"Encontrada carpeta CFDNA_DIR={CFDNA_DIR}")

    if not os.path.exists(GENCODE_DIR):
        os.mkdir(GENCODE_DIR)
        # raise ValueError("No encontrada ruta",GENCODE_DIR)
    else:
        verbose(f"Encontrada carpeta GENCODE_DIR={GENCODE_DIR}")

    if not os.path.exists(H5_DIR):
        os.mkdir(H5_DIR)
        verbose(f"Creada carpeta H5_DIR={H5_DIR}")
    else:
        verbose(f"Encontrada carpeta H5_DIR={H5_DIR}")

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
        verbose(f"Creada carpeta LOG_DIR={LOG_DIR}")
    else:
        verbose(f"Encontrada carpeta LOG_DIR={LOG_DIR}")

    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
        verbose(f"Creada carpeta CSV_DIR={CSV_DIR}")
    else:
        verbose(f"Encontrada carpeta CSV_DIR={CSV_DIR}")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        verbose(f"Creada carpeta MODEL_DIR={MODEL_DIR}")
    else:
        verbose(f"Encontrada carpeta MODEL_DIR={MODEL_DIR}")

    if not os.path.exists(EXEC_DIR):
        os.makedirs(EXEC_DIR)
        verbose(f"Creada carpeta EXEC_DIR={EXEC_DIR}")
    else:
        verbose(f"Encontrada carpeta EXEC_DIR={EXEC_DIR}")

    if not os.path.exists(MET_DIR):
        os.makedirs(MET_DIR)
        verbose(f"Creada carpeta MET_DIR={MET_DIR}")
    else:
        verbose(f"Encontrada carpeta MET_DIR={MET_DIR}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# limpiar de memoria una variable local o global una vez que no sea necesaria.
def del_local_var_bk(variables):
    for var1 in variables:
        if var1 in locals():
            del locals()[var1]
            print (f"Borrada de las variables locales {var1}")
        else:
            if var1 in globals():
                del globals()[var1]
                print (f"Borrada de las variables globales {var1}")
            else:
                print (f"NO encontrada entre las variables globales ni locales {var1}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Borrar fichero si existe
def del_file_if_exists(fichero1):
    if os.path.exists(fichero1): 
        os.remove(fichero1)
        verbose(f"Fichero borrado {fichero1}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# leer fichero de datos cfdna
def read_data_file(fichero1, carpeta1=CFDNA_DIR, nfilas1=None):
    fichero2 = os.path.join(carpeta1,fichero1)
    if not os.path.exists(fichero2):
        raise ValueError(f"Fichero no encontrado {fichero2}")

    df1 = pd.read_csv(fichero2, sep='\t', nrows = nfilas1)

    if "Geneid" in df1.columns:
        df1 = df1.rename(columns = {"Geneid": "id"})

    return df1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# leer fichero de gencode con las columnas necesarias
def read_gtf_file(fichero1, carpeta1=GENCODE_DIR, nfilas1=None):
    fichero2 = os.path.join(carpeta1,fichero1)
    if not os.path.exists(fichero2):
        raise ValueError(f"Fichero no encontrado {fichero2}")

    columnas = ['chromosome_name', 'annotation_source', 'feature_type', 'genomic_start_location', 'genomic_end_location', 'score', 'genomic_strand', 'genomic_phase', 'attributes']

    df1 = pd.read_csv(fichero2, sep='\t', nrows = nfilas1, header=None, names=columnas)

    return df1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# añadir al final de las columnas, las nuevas columnas de las muestras para el tipo de cáncer(target1)
def add_target_row(df_tmp, df_c, target1, union_columnas):

    classes_row = pd.DataFrame([[target1] * (len(df_tmp.columns)-len(union_columnas))], columns = df_tmp.columns[len(union_columnas):])
    
    join_row = pd.DataFrame([["target","target","1","1","+","1"]], columns = union_columnas)
    
    nueva_fila  = pd.concat([join_row, classes_row], axis = 1)
    
    if df_c.empty:
        df_c = nueva_fila.copy()
    else:
        df_c = df_c.merge(nueva_fila, on = union_columnas, how = "outer")
    return df_c


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar el CSV en formato español para abrirlo en excel
def save_df_to_csv_spa(df_bk, fichero1, carpeta1=CSV_DIR):

    save_df_to_csv(df_bk, fichero1, carpeta1, separa1=";", decima1=",", guarda_indice=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Leer el CSV en formato español para abrirlo en excel
def read_csv_to_df_spa(fichero1, carpeta1=CSV_DIR):

    return read_csv_to_df(fichero1, carpeta1, separa1=";", decima1=",", columna_indice=0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar el CSV en formato por defecto
def save_df_to_csv(df_bk, fichero1, carpeta1=CSV_DIR, separa1=",", decima1=".", guarda_indice=False):

    if os.path.splitext(fichero1)[-1].lower() != ".csv": fichero1 = fichero1+".csv"

    verbose(f"Inicio guardar fichero {os.path.join(carpeta1,fichero1)}")
    
    del_file_if_exists(os.path.join(carpeta1, fichero1))
    
    df_bk.to_csv(os.path.join(carpeta1, fichero1), index=guarda_indice, sep=separa1, decimal=decima1)
    
    verbose(f"Fin guardar fichero {os.path.join(carpeta1,fichero1)}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Leer el CSV en formato por defecto
def read_csv_to_df(fichero1, carpeta1=CSV_DIR, separa1=",", decima1=".", columna_indice=None):

    if os.path.splitext(fichero1)[-1].lower() != ".csv": fichero1 = fichero1+".csv"
    
    verbose(f"Inicio leer fichero {os.path.join(carpeta1,fichero1)}")
    
    if not os.path.exists(os.path.join(carpeta1, fichero1)):
        raise ValueError(f"Fichero no encontrado {os.path.join(carpeta1, fichero1)}")
    
    df_r = pd.read_csv(os.path.join(carpeta1, fichero1), sep=separa1, decimal=decima1, index_col=columna_indice)
    
    verbose(f"Fin leer fichero {os.path.join(carpeta1,fichero1)}")
    
    return df_r


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar el DataFrame en formato HDF con la clave por defecto key1=df
def save_df_to_h5(df_bk, fichero1, carpeta1=H5_DIR, key1="df"):

    if os.path.splitext(fichero1)[-1].lower() != ".h5": fichero1 = fichero1+".h5"

    verbose(f"Inicio guardar fichero h5 {os.path.join(carpeta1,fichero1)} con clave {key1}")

    del_file_if_exists(os.path.join(carpeta1, fichero1))

    df_bk.to_hdf(os.path.join(carpeta1, fichero1), key=key1, mode='w')

    verbose(f"Fin guardar fichero h5 {os.path.join(carpeta1,fichero1)} con clave {key1}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar el DataFrame en el fichero HDF con la clave key1
def save_df_to_key_h5(df_bk, fichero1, carpeta1=H5_DIR, key1="df"):

    if os.path.splitext(fichero1)[-1].lower() != ".h5": fichero1 = fichero1+".h5"

    verbose(f"Inicio guardar fichero h5 {os.path.join(carpeta1,fichero1)} con clave {key1}")

    if key1 == "df": del_file_if_exists(os.path.join(carpeta1, fichero1))

    df_bk.to_hdf(os.path.join(carpeta1, fichero1), key=key1, mode='a')

    verbose(f"Fin guardar fichero h5 {os.path.join(carpeta1,fichero1)} con clave {key1}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Leer el fichero HDF y devolver un DataFrame
def read_h5_to_df(fichero1, carpeta1=H5_DIR):

    if os.path.splitext(fichero1)[-1].lower() != ".h5": fichero1 = fichero1+".h5"
    
    verbose(f"Inicio leer fichero h5 {os.path.join(carpeta1,fichero1)}")

    if not os.path.exists(os.path.join(carpeta1, fichero1)):
        raise ValueError(f"Fichero no encontrado {os.path.join(carpeta1, fichero1)}")

    df_cc = pd.DataFrame()

    with pd.HDFStore(os.path.join(carpeta1,fichero1)) as store:
        las_keys=store.keys()

    for key1 in las_keys:
        verbose(f"Leyendo clave {key1}")
        df_r = pd.read_hdf(os.path.join(carpeta1, fichero1), key=key1)

        if df_cc.empty:
            df_cc = df_r.copy()
        else:
            df_cc  = pd.concat([df_cc, df_r], axis = 0)

        verbose(f"Tamaño {df_cc.shape}")

    verbose(f"Fin leer fichero h5 {os.path.join(carpeta1,fichero1)}")

    return df_cc


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Listar las claves del fichero HDF
def list_h5_keys(fichero1, carpeta1=H5_DIR):

    if os.path.splitext(fichero1)[-1].lower() != ".h5": fichero1 = fichero1+".h5"
    
    verbose(f"Inicio listar h5 claves en fichero {os.path.join(carpeta1,fichero1)}")

    with pd.HDFStore(os.path.join(carpeta1,fichero1)) as store:
        las_keys=store.keys()

    for key1 in las_keys:
        verbose(f"Encontrada clave {key1}")

    verbose(f"Fin listar h5 claves en fichero {os.path.join(carpeta1,fichero1)}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Guardar los resultados de los distintos entrenamiento hecho con el RandomizedSearchCV en un CSV
def save_resultados_to_csv(tipo, select, clasific, search, carpeta1=MET_DIR, fichero1="resultados.csv"):

    verbose(f"Inicio guardar resultados")

    if os.path.splitext(fichero1)[-1].lower() != ".csv": fichero1 = fichero1+".csv"

    if os.path.exists(os.path.join(carpeta1,fichero1)):
        df_m=read_csv_to_df_spa(fichero1, carpeta1)
    else:
        df_m=pd.DataFrame()

    df_new = pd.DataFrame(columns=["tipo","select","clasific","hostname","datetime"])
    df_new=pd.concat([df_new, pd.DataFrame(search.cv_results_)], axis = 0)
    df_new["tipo"] = str.replace(tipo," ", "_")
    df_new["select"] = str.replace(select," ", "_")
    df_new["clasific"] = str.replace(clasific," ", "_")
    df_new["hostname"] = platform.node()
    df_new["datetime"] = datetime.datetime.now()

    save_df_to_csv_spa(pd.concat([df_m, df_new], axis = 0), fichero1, carpeta1)

    verbose(f"Fin guardar resultados")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Ver los resultados guardados en un CSV
def ver_resultados_search(search):

    verbose(f"Inicio ver resultados")

    df_res1=pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)

    print(f'Mejores Hiperparámetros: {search.best_params_}')

    for mean_score, params in zip(df_res1[df_res1.rank_test_score==1]["mean_test_score"], df_res1[df_res1.rank_test_score==1]["params"]):
        print(f'Mejores puntuaciones: {mean_score:.4f} | Params: {params}')

    for mean_score, params in zip(df_res1["mean_test_score"], df_res1["params"]):
        print(f'Media de las puntuaciones: {mean_score:.4f} | Params: {params}')

    verbose(f"Fin resultados")



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver matriz de confusión
def ver_metricas_multi_matriz_confusion(y_test, y_pred):

    print("\n")

    l_classes = np.unique(y_test)

    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=l_classes, yticklabels=l_classes)
    plt.xlabel('Predicho')
    plt.ylabel('Actual',)
    plt.title('Matriz de Confusión')
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver curva ROC ACU
def ver_metricas_multi_roc_auc(y_test, y_pred_proba):

    print("\n")

    n_classes = len(np.unique(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc_multi = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc_multi[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        print(f"AUC para el tipo de cáncer {i}-{TARGETS[i].ljust(11, '.')} {roc_auc_multi[i]:.4f}")

    roc_auc_media = sklearn.metrics.roc_auc_score(y_test, y_pred_proba, average="macro", multi_class="ovr")
    print(f"\nMedia de AUC: {roc_auc_media:.4f}")

    plt.figure(figsize=(10, 6))
    colors = ['green', 'blue', 'purple', 'red', 'black', 'orange', 'gray']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"AUC para tipo de cáncer {i}-{TARGETS[i]}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Linea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio Falso Positivo (FPR)')
    plt.ylabel('Ratio Verdadero Positivo (TPR)')
    plt.title('AUC para multiclase')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc_multi


def metricas_roc_auc(y_test, y_pred_proba):

    n_classes = len(np.unique(y_test))

    fpr = dict()
    tpr = dict()
    roc_auc_multi = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc_multi[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    return roc_auc_multi


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver metricas Kappa
def ver_metricas_multi_kappa(y_test, y_pred):

    print("\n")

    n_classes = len(np.unique(y_test))

    kappa_scores = []

    y_test_binarized = sklearn.preprocessing.label_binarize(y_test, classes=list(range(n_classes)))
    y_pred_binarized = sklearn.preprocessing.label_binarize(y_pred, classes=list(range(n_classes)))

    for i in range(n_classes):
        kappa = sklearn.metrics.cohen_kappa_score(y_test_binarized[:, i], y_pred_binarized[:, i])
        kappa_scores.append(kappa)
        print(f"Puntuación Kappa Cohen (OvR) para la clase {i}-{TARGETS[i].ljust(11, '.')} {kappa:.4f}")

    avg_kappa = sum(kappa_scores) / n_classes
    print(f"\nMedia de la puntuación Kappa Cohen (OvR): {avg_kappa:.4f}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver metricas Gini
def ver_metricas_multi_gini(n_classes, roc_auc):

    print("\n")

    gini_scores = []

    for i in range(n_classes):
        gini = 2 * roc_auc[i] - 1
        gini_scores.append(gini)
        print(f"Puntuación de Gini para el tipo de cáncer {i}-{TARGETS[i].ljust(11, '.')}: {gini:.4f}")

    avg_gini = sum(gini_scores) / n_classes
    print(f"\nMedia de la puntuación de Gini (OvR): {avg_gini:.4f}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver metricas jaccard
def ver_metricas_multi_jaccard(y_test, y_pred):

    n_classes = len(np.unique(y_test))

    print("\n")

    jaccard_per_class = metricas_jaccard(y_test, y_pred)

    for i in range(n_classes):
        print(f"Puntuación Jaccard para el cáncer tipo {i}-{TARGETS[i].ljust(11, '.')} {jaccard_per_class[i]:.4f}")

    print(f"\nMedia de la puntuación de Jaccard: {np.mean(jaccard_per_class):.4f}")


def metricas_jaccard(y_test, y_pred):

    def jaccard_index(true, pred):
        intersection = np.logical_and(true, pred)
        union = np.logical_or(true, pred)
        return intersection.sum() / float(union.sum())

    n_classes = len(np.unique(y_test))

    y_test_binarized = sklearn.preprocessing.label_binarize(y_test, classes=list(range(n_classes)))
    y_pred_binarized = sklearn.preprocessing.label_binarize(y_pred, classes=list(range(n_classes)))

    jaccard_per_class = []

    for class_val in np.unique(y_test):
        true_class = np.array(y_test) == class_val
        pred_class = np.array(y_pred) == class_val
        jaccard_per_class.append(jaccard_index(true_class, pred_class))

    true_one_hot = np.zeros((len(y_test), len(np.unique(y_test))))
    pred_one_hot = np.zeros_like(true_one_hot)

    for i, val in enumerate(y_test):
        true_one_hot[i, val] = 1

    for i, val in enumerate(y_pred):
        pred_one_hot[i, val] = 1

    return jaccard_per_class


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver matriz de confusión binaria
def ver_metricas_bin_matriz_confusion(y_test, y_pred):

    l_classes = np.unique(y_test)
    
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(3, 1))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=l_classes, yticklabels=l_classes)
    plt.xlabel('Predicho')
    plt.ylabel('Real',)
    plt.title('Matriz de Confusión')
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver matriz de confusión binaria
def ver_metricas_bin_roc_auc(y_test, y_pred_proba):

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Linea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio Falso Positivo (FPR)')
    plt.ylabel('Ratio Verdadero Positivo (TPR)')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver todas las metricas obtenidas par un modelo de clasificación binario
def ver_metricas(modelo, X_test, y_test):
    verbose(f"Inicio ver métricas")

    y_pred1 = modelo.predict(X_test)

    # Para el caso de que no sea solo 0 y 1, como la función sigmoid que saca valores entre 0 y 1
    umbral = 0.5 
    y_pred = (y_pred1 > umbral).astype(int)
    y_pred_proba = modelo.predict_proba(X_test)[:,1]

    print(sklearn.metrics.classification_report(y_test, y_pred, zero_division=0, digits=3))

    conf_matrix=sklearn.metrics.confusion_matrix(y_test, y_pred)

    accuracy=sklearn.metrics.accuracy_score(y_test, y_pred)
    precision=sklearn.metrics.precision_score(y_test, y_pred, average="macro")
    recall=sklearn.metrics.recall_score(y_test, y_pred, average="macro")
    f1=sklearn.metrics.f1_score(y_test, y_pred, average="macro")
    roc_auc=sklearn.metrics.roc_auc_score(y_test, y_pred_proba)

    print("\n")
    print("   Exactitud: %.4f" % (accuracy)) 
    print("   Precisión: %.4f" % (precision))
    print("Sensibilidad: %.4f" % (recall))
    print("    F1-score: %.4f" % (f1))
    print("     AUC ROC: %.4f" % (roc_auc))
    
    ver_metricas_bin_matriz_confusion(y_test,y_pred)
    
    ver_metricas_bin_roc_auc(y_test, y_pred_proba)

    verbose(f"Fin ver métricas")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver todas las metricas obtenidas par un modelo de clasificación multiclase
def ver_metricas_multi(modelo, X_test, y_test):
    verbose(f"Inicio ver métricas multiclase")

    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)

    print(sklearn.metrics.classification_report(y_test, y_pred, zero_division=0, digits=3))

    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_test, y_pred, average="macro")
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average="macro")

    print("\n")
    print("           Exactitud: %.6f" % (accuracy)) 
    print("   Precisión (media): %.6f" % (precision))
    print("      Recall (media): %.6f" % (recall))
    print("    F1-score (media): %.6f" % (f1))


    roc_auc_multi=ver_metricas_multi_roc_auc(y_test, y_pred_proba)

    ver_metricas_multi_matriz_confusion(y_test,y_pred)

    # ver_metricas_multi_kappa(y_test, y_pred)

    # n_classes = len(np.unique(y_test))
    # ver_metricas_multi_gini(n_classes, roc_auc_multi)

    ver_metricas_multi_jaccard(y_test, y_pred)
 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Ver todas las metricas obtenidas por un modelo de clasificación multiclase del pipeline
def ver_metricas_multi_pipeline(modelo_bin, modelo_mul, X_test, y_test):
    def predict_pipelines(modelo_bin, modelo_mul, X):
        predict_bin  = modelo_bin.predict(X)
        predict_mul  = modelo_mul.predict(X)
        return np.where(predict_bin == 0, 0, predict_mul)

    def predict_proba_pipelines(modelo_bin, modelo_mul, X):
        predict_proba_bin2  = modelo_bin.predict_proba(X)
        predict_proba_bin = np.pad(predict_proba_bin2,  ((0, 0), (0, 5)), 'constant')
        predict_proba_mul  = modelo_mul.predict_proba(X)
        mask = predict_proba_bin[:, 0] > 0.50
        return np.where(mask[:, np.newaxis], predict_proba_bin, predict_proba_mul)

    y_pred = predict_pipelines(modelo_bin, modelo_mul, X_test)
    y_pred_proba = predict_proba_pipelines(modelo_bin, modelo_mul, X_test)

    print(sklearn.metrics.classification_report(y_test, y_pred, zero_division=0, digits=3))

    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_test, y_pred, average="macro")
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average="macro")

    print("\n")
    print("           Exactitud: %.6f" % (accuracy)) 
    print("   Precisión (media): %.6f" % (precision))
    print("      Recall (media): %.6f" % (recall))
    print("    F1-score (media): %.6f" % (f1))

    n_classes = len(np.unique(y_test))

    roc_auc_multi=ver_metricas_multi_roc_auc(y_test, y_pred_proba)

    ver_metricas_multi_matriz_confusion(y_test,y_pred)

    # ver_metricas_multi_kappa(y_test, y_pred)

    # ver_metricas_multi_gini(n_classes, roc_auc_multi)

    ver_metricas_multi_jaccard(y_test, y_pred)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Leer el fichero CSV con las métricas
def read_metricas_to_df(carpeta1=MET_DIR, fichero1="metricas.csv"):
    return read_csv_to_df_spa(fichero1, carpeta1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar en un fichero CSV las métricas de un clasificador binario
def save_metricas_to_csv(modelos, X_test, y_test, tipo, select, clasific, total_time, semilla=42, carpeta1=MET_DIR, fichero1="metricas.csv", fichero_modelo="modelo.pkl"):
    
    mejor_modelo = modelos.best_estimator_
    parametros = modelos.best_params_
    mascara_columnas = modelos.best_estimator_.named_steps['selector'].get_support()
    columnas1 = list(X_test.columns[mascara_columnas])

    verbose(f"Inicio guardar métricas")
    if os.path.splitext(fichero1)[-1].lower() != ".csv": fichero1 = fichero1+".csv"
    if os.path.exists(os.path.join(carpeta1,fichero1)):
        df_m=read_csv_to_df_spa(fichero1, carpeta1)
    else:
        df_m=pd.DataFrame()

    y_pred_prob = modelos.predict(X_test)

    umbral = 0.5 
    y_pred = (y_pred_prob > umbral).astype(int)
    y_pred_proba = modelos.predict_proba(X_test)[:,1]

    accuracy=sklearn.metrics.accuracy_score(y_test, y_pred)
    precision=sklearn.metrics.precision_score(y_test, y_pred, average="macro")
    recall=sklearn.metrics.recall_score(y_test, y_pred, average="macro")
    f1=sklearn.metrics.f1_score(y_test, y_pred, average="macro")
    roc_auc=sklearn.metrics.roc_auc_score(y_test, y_pred_proba)

    columnas = ['tipo', 'select', 'clasific', 'semilla', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'hostname', 'total_time_sec', 'datetime', 'fichero_modelo', 'params', 'shape','features']
    df_new = pd.DataFrame(columns=columnas, index=[0])
    df_new["tipo"] = str.replace(tipo," ", "_")
    df_new["select"] = str.replace(select," ", "_")
    df_new["clasific"] = str.replace(clasific," ", "_")
    df_new["accuracy"] = accuracy
    df_new["precision"] = precision
    df_new["recall"] = recall
    df_new["f1_score"] = f1
    df_new["roc_auc"] = roc_auc
    df_new["hostname"] = platform.node()
    df_new["datetime"] = datetime.datetime.now()
    df_new["total_time_sec"] = total_time.seconds
    df_new["semilla"] = semilla
    df_new["fichero_modelo"] = fichero_modelo
    df_new["params"] = str(parametros)
    df_new["shape"] = "X_test:"+str(X_test.shape)
    df_new["features"] = str(columnas1)

    save_df_to_csv_spa(pd.concat([df_m, df_new], axis = 0), fichero1, carpeta1)

    verbose(f"Fin guardar métricas")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar en un fichero CSV las métricas de un clasificador multiclase
def save_metricas_multi_to_csv(modelos, X_test, y_test, tipo, select, clasific, total_time, semilla=42, carpeta1=MET_DIR, fichero1="metricas.csv", fichero_modelo="modelo.pkl"):
    
    if "best_estimator_" in modelos.__dict__:
        mejor_modelo = modelos.best_estimator_
        parametros = modelos.best_params_
        mascara_columnas = modelos.best_estimator_.named_steps['selector'].get_support()
        columnas1 = list(X_test.columns[mascara_columnas])
    else:
        mejor_modelo = ""
        parametros = modelos.steps
        mascara_columnas = ""
        columnas1 = ""
    

    verbose(f"Inicio guardar métricas multiclase")
    if os.path.splitext(fichero1)[-1].lower() != ".csv": fichero1 = fichero1+".csv"
    if os.path.exists(os.path.join(carpeta1,fichero1)):
        df_m=read_csv_to_df_spa(fichero1, carpeta1)
    else:
        df_m=pd.DataFrame()

    y_pred = modelos.predict(X_test)
    y_pred_proba = modelos.predict_proba(X_test)

    accuracy=sklearn.metrics.accuracy_score(y_test, y_pred)
    precision=sklearn.metrics.precision_score(y_test, y_pred, average="macro")
    recall=sklearn.metrics.recall_score(y_test, y_pred, average="macro")
    f1=sklearn.metrics.f1_score(y_test, y_pred, average="macro")
    roc_auc_ovr=sklearn.metrics.roc_auc_score(y_test, y_pred_proba, average="macro", multi_class="ovr")

    columnas = ['tipo', 'select', 'clasific', 'semilla', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_ovr', 'indices_auc','indices_jaccard', 'hostname', 'total_time_sec', 'datetime', 'fichero_modelo', 'params', 'shape','features']
    df_new = pd.DataFrame(columns=columnas, index=[0])
    df_new["tipo"] = str.replace(tipo," ", "_")
    df_new["select"] = str.replace(select," ", "_")
    df_new["clasific"] = str.replace(clasific," ", "_")
    df_new["accuracy"] = accuracy
    df_new["precision"] = precision
    df_new["recall"] = recall
    df_new["f1_score"] = f1
    df_new["roc_auc_ovr"] = roc_auc_ovr
    df_new["indices_auc"] = str(list(metricas_roc_auc(y_test, y_pred_proba).values()))
    df_new["indices_jaccard"] = str(metricas_jaccard(y_test, y_pred))
    df_new["hostname"] = platform.node()
    df_new["datetime"] = datetime.datetime.now()
    df_new["total_time_sec"] = total_time.seconds
    df_new["semilla"] = semilla
    df_new["fichero_modelo"] = fichero_modelo
    df_new["params"] = str(parametros)
    df_new["shape"] = "X_test:"+str(X_test.shape)
    df_new["features"] = str(columnas1)

    save_df_to_csv_spa(pd.concat([df_m, df_new], axis = 0), fichero1, carpeta1)

    verbose(f"Fin guardar métricas multiclase")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Guardar el modelo en un pkl
def save_modelo(modelo1, carpeta1=MODEL_DIR, fichero1="modelo.pkl"):
    if os.path.splitext(fichero1)[-1].lower() != ".pkl": fichero1 = fichero1+".pkl"
    verbose(f"Inicio guardar fichero {os.path.join(carpeta1,fichero1)}")
    del_file_if_exists(os.path.join(carpeta1, fichero1))
    joblib.dump(modelo1, os.path.join(carpeta1, fichero1))
    verbose(f"Fin guardar fichero {os.path.join(carpeta1,fichero1)}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Leer el modelo de un pkl
def read_modelo(carpeta1=MODEL_DIR, fichero1="modelo.pkl"):
    if os.path.splitext(fichero1)[-1].lower() != ".pkl": fichero1 = fichero1+".pkl"
    verbose(f"Inicio leer fichero {os.path.join(carpeta1,fichero1)}")
    
    if not os.path.exists(os.path.join(carpeta1, fichero1)):
        raise ValueError(f"Fichero no encontrado {os.path.join(carpeta1, fichero1)}")
    
    modelo= joblib.load(os.path.join(carpeta1,fichero1))
    verbose(f"Fin leer fichero {os.path.join(carpeta1,fichero1)}")
    return modelo


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Poner el inicio de un punto de chequeo para medir el tiempo
def start_point():
    global START_CHKPOINT
    global LAST_CHKPOINT
    global END_CHKPOINT
    START_CHKPOINT = datetime.datetime.now()
    LAST_CHKPOINT = datetime.datetime.now()
    verbose(f"Iniciando contador a las {START_CHKPOINT}")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Establecer un nuevo punto de chequeo y mostrar el tiempo pasado desde el último punto
def chk_point():
    global START_CHKPOINT
    global LAST_CHKPOINT
    global END_CHKPOINT
    NOW_CHKPOINT = datetime.datetime.now()
    time_since_last_chkpoint = NOW_CHKPOINT - LAST_CHKPOINT
    total_time = NOW_CHKPOINT - START_CHKPOINT
    verbose(f"Tiempo desde último chkpoint {time_since_last_chkpoint} segundos, desde el inicio {total_time} segundos")
    LAST_CHKPOINT = NOW_CHKPOINT


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Mostrar las versiones actuales de los modulos utilidados
def versions():
    verbose(f"Versión {sys.__name__} {sys.version}")
    verbose(f"Versión {np.__name__} {np.__version__}")
    verbose(f"Versión {pd.__name__} {pd.__version__}")
    verbose(f"Versión {platform.__name__} {platform.__version__}")    
    verbose(f"Versión {psutil.__name__} {psutil.__version__}")
    verbose(f"Versión {sklearn.__name__} {sklearn.__version__}")
    verbose(f"Versión {PIL.__name__} {PIL.__version__}")
    verbose(f"Versión {matplotlib.__name__} {matplotlib.__version__}")
    verbose(f"Versión {sns.__name__} {sns.__version__}")
    verbose(f"Versión {IPython.__name__} {IPython.__version__}")
    verbose(f"Versión {logging.__name__} {logging.__version__}")
    verbose(f"Versión {joblib.__name__} {joblib.__version__}")
