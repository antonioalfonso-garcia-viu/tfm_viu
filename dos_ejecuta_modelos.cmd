@echo off

set TFM_DIR=%cd%
@REM echo Carpeta de los fuentes %TFM_DIR%

if "%1"=="" (
    set EJEC_DIR=ejecuciones
) else (
    set EJEC_DIR=%1
)

set PARAM_DIR=%TFM_DIR%\%EJEC_DIR%\parametros

if NOT EXIST %PARAM_DIR% (
    echo "NOTEBOOK:ERROR:%ahora%:dos_ejecuta_modelos.cmd:Carpeta %PARAM_DIR% no existe"
    echo.&goto:eof
)

cd %PARAM_DIR%

if NOT EXIST %PARAM_DIR%\done mkdir %PARAM_DIR%\done
if NOT EXIST %PARAM_DIR%\error mkdir %PARAM_DIR%\error

for %%a in (*.yaml) do call:myDosFunc %%a
cd %TFM_DIR%

@REM ==================funcion==================

echo.&goto:eof

:myDosFunc

cd %TFM_DIR%

set PARAM1=%~n1
set TRAIN1=03_entrenamiento
set fahora=%DATE:~6,9%%DATE:~3,2%%DATE:~0,2%T%time:~0,2%%time:~3,2%%time:~6,2%
set ahora=%fahora: =_%

echo.
IF NOT EXIST %TRAIN1%.ipynb (
    echo NOTEBOOK:ERROR:%ahora%:dos_ejecuta_modelos.cmd:Archivo %TRAIN1% no existe
) else (
    IF NOT EXIST %PARAM_DIR%\%PARAM1%.yaml (
        echo NOTEBOOK:ERROR:%ahora%:dos_ejecuta_modelos.cmd:Archivo parametros_final\%PARAM1%.yaml no existe
    ) else (
        echo.
        echo NOTEBOOK:INFO:%ahora%:dos_ejecuta_modelos.cmd:Inicio ejecucion call dos_ejecuta1modelo.cmd %EJEC_DIR% %TRAIN1% %PARAM1%
        call dos_ejecuta1modelo.cmd %EJEC_DIR% %TRAIN1% %PARAM1%
        echo.
    )
)

goto:eof
