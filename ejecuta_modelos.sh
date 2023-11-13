#!/bin/bash

[[ -f "$HOME/.bash_profile" ]] && . $HOME/.bash_profile

TFM_DIR="$(dirname "$0")"
cd "${TFM_DIR}"
TFM_DIR="$(pwd)"
 
if [[ ! -d "${TFM_DIR}" ]]
then
    echo "Problema al acceder a laaa carpeta ${TFM_DIR}" 
    exit -1
fi


ejecuta1() {
    for PARAM1 in ${LISTA1}; do
        export PARAM1=${PARAM1%.*}
        export TRAIN1=03_entrenamiento
        export ahora="$(date +%Y%m%dT%H%M%S)"
        [[ ! -f "${TRAIN1}.ipynb" ]] && echo "No encuentro ${TRAIN1}.ipynb" && exit 64
        [[ ! -f "${PARAM_DIR}/${PARAM1}.yaml" ]] && echo "No encuentro ${PARAM_DIR}/${PARAM1}.yaml" && exit 65
        echo "NOTEBOOK:INFO:${ahora}:ejecuta_modelos.sh:Inicio ejecucion bash ejecuta1modelo.sh ${TRAIN1} ${PARAM1}"
        bash ejecuta1modelo.sh "${EJEC_DIR}" "${TRAIN1}" "${PARAM1}"
        sleep 1
    done
}

if [[ -z $1 ]]
then
    EJEC_DIR=ejecuciones
else
    EJEC_DIR=$1
fi

export PARAM_DIR=${TFM_DIR}/${EJEC_DIR}/parametros

if [[ ! -d "${PARAM_DIR}" ]]
then
    echo "Problema al acceder a laaa carpeta ${PARAM_DIR}" 
    exit -1
else
    mkdir -p ${PARAM_DIR}/done
    mkdir -p ${PARAM_DIR}/error
fi

cd "${PARAM_DIR}"

LISTA1=$(ls *.yaml)

cd "${TFM_DIR}"

ejecuta1 "${LISTA1}"
