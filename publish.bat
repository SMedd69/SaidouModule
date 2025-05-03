@echo off
echo ------------------------------
echo  Suppression des anciens fichiers...
echo ------------------------------
rmdir /s /q build
rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q %%i

echo ------------------------------
echo  Construction du package...
echo ------------------------------
python -m build

if %errorlevel% neq 0 (
    echo Échec de la construction du package.
    exit /b %errorlevel%
)

echo ------------------------------
echo  Téléversement vers PyPI...
echo ------------------------------
python -m twine upload dist/*

if %errorlevel% neq 0 (
    echo Échec du téléversement.
    exit /b %errorlevel%
)

echo ------------------------------
echo  ✅ Publication terminée !
echo ------------------------------
pause
