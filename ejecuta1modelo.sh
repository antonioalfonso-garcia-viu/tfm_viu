#!/bin/bash
export EJEC_DIR=$1
export FILE1=$2
export PARAM1=$3

echo .
echo ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
echo .

export PARAM_DIR=${EJEC_DIR}/parametros
export IPYNB_FILE=${FILE1}.ipynb
export OUT_FILE1=${EJEC_DIR}/training/${FILE1}.${ahora}.${PARAM1}.output.ipynb
export LOG_FILE1=${EJEC_DIR}/training/${FILE1}.${ahora}.${PARAM1}.output.ipynb.log
export PARAM_FILE1=${PARAM_DIR}/${PARAM1}.yaml

export ahora="$(date +%Y%m%dT%H%M%S)"
echo "NOTEBOOK:INFO:${ahora}:${IPYNB_FILE}:Iniciado con $*"

# OTROS_ARGS="--execution-timeout 5"
OTROS_ARGS=""
if papermill "${IPYNB_FILE}" "${OUT_FILE1}" -f "${PARAM_FILE1}" ${OTROS_ARGS} >"${LOG_FILE1}" 2>&1; then
    mv "${PARAM_DIR}/${PARAM1}.yaml" "${PARAM_DIR}/done/"
    echo "NOTEBOOK:INFO:${ahora}:${IPYNB_FILE}:Movido ${PARAM_DIR}/${PARAM1}.yaml a DONE"
else
    mv "${PARAM_DIR}/${PARAM1}.yaml" "${PARAM_DIR}/error/"
    echo "NOTEBOOK:ERROR:${ahora}:${IPYNB_FILE}:Movido ${PARAM_DIR}/${PARAM1}.yaml a ERROR"

    mv "${OUT_FILE1}" "${OUT_FILE1}.ERROR.ipynb"
    echo "NOTEBOOK:ERROR:${ahora}:${IPYNB_FILE}:Renombrado ${OUT_FILE1}.ERROR.ipynb"

    mv "${LOG_FILE1}" "${LOG_FILE1}.ERROR.log"
    echo "NOTEBOOK:ERROR:${ahora}:${IPYNB_FILE}:Renombrado ${LOG_FILE1}.ERROR.log"
fi

export ahora="$(date +%Y%m%dT%H%M%S)"
echo "NOTEBOOK:INFO:${ahora}:${IPYNB_FILE}:Finalizado"
echo .