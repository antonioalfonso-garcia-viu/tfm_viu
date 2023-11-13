@echo off

set EJEC_DIR=%1
set FILE1=%2
set PARAM1=%3

echo.
echo ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
echo.

set PARAM_DIR=%EJEC_DIR%\parametros

if NOT EXIST %PARAM_DIR% (
    echo NOTEBOOK:ERROR:%ahora%:dos_ejecuta1modelo.cmd:Carpeta %PARAM_DIR% no existe
    echo.&goto:eof
)

set IPYNB_FILE=%FILE1%.ipynb
set OUT_FILE1=%EJEC_DIR%\training\%FILE1%.%ahora%.%PARAM1%.output.ipynb
set PARAM_FILE1=%PARAM_DIR%\%PARAM1%.yaml
set LOG_FILE1="%OUT_FILE1%.log"

set fahora=%DATE:~6,9%%DATE:~3,2%%DATE:~0,2%T%time:~0,2%%time:~3,2%%time:~6,2%
set ahora=%fahora: =_%
echo.
echo NOTEBOOK:INFO:%ahora%:dos_modelo1ejefinal.cmd:Iniciado con %*

@REM set OTROS_ARGS=--execution-timeout 5
set OTROS_ARGS=

papermill.exe "%IPYNB_FILE%" "%OUT_FILE1%" -f "%PARAM_FILE1%" %OTROS_ARGS% > "%LOG_FILE1%" 2>&1

IF %ERRORLEVEL% EQU 0 (
    set fahora=%DATE:~6,9%%DATE:~3,2%%DATE:~0,2%T%time:~0,2%%time:~3,2%%time:~6,2%
    set ahora=%fahora: =_%
    echo NOTEBOOK:INFO:%ahora%:dos_modelo1ejefinal.cmd:Movido %PARAM_DIR%\%PARAM1%.yaml a DONE
    move "%PARAM_DIR%\%PARAM1%.yaml" "%PARAM_DIR%\done\"
) else (
    set fahora=%DATE:~6,9%%DATE:~3,2%%DATE:~0,2%T%time:~0,2%%time:~3,2%%time:~6,2%
    set ahora=%fahora: =_%
    move "%PARAM_DIR%\%PARAM1%.yaml" "%PARAM_DIR%\error\"
    echo NOTEBOOK:INFO:%ahora%:dos_modelo1ejefinal.cmd:Movido %PARAM_DIR%\%PARAM1%.yaml a ERROR
    move "%OUT_FILE1%"  "%OUT_FILE1%.ERROR.ipynb"
    echo NOTEBOOK:INFO:%ahora%:dos_modelo1ejefinal.cmd:Renombrado %OUT_FILE1%.ERROR.ipynb
    move "%LOG_FILE1%"  "%LOG_FILE1%.ERROR.log"
    echo NOTEBOOK:INFO:%ahora%:dos_modelo1ejefinal.cmd:Renombrado %LOG_FILE1%.ERROR.log
)
echo NOTEBOOK:INFO:%ahora%:dos_modelo1ejefinal.cmd:Finalizado
echo.

goto:eof