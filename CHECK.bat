@echo off
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
echo.
echo === Environment check ===
python -c "import numpy, scipy, networkx, pyscipopt; print('  core OK')"
if errorlevel 1 goto FAIL
python -c "import neal; print('  dwave-neal OK')"
if errorlevel 1 echo   [warn] dwave-neal missing: pip install dwave-neal
python qaoa_gpu.py
echo.
echo === Quick FMQA test (about 1 min) ===
python fmqa.py --k 3 --opt exact --n-init 40 --n-iter 60 --repeats 1 --workers 20
goto END
:FAIL
echo   FAILED. pip install pyscipopt networkx numpy scipy dwave-neal
:END
pause
