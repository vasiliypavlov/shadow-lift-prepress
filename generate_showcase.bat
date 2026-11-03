@echo off
setlocal

:: Имя основного скрипта
set SCRIPT=shadow_lift_depth_shadow_prepress.py
:: Входной файл
set INPUT=docs/showcase_input.jpg
:: Папка для выходных файлов
set OUT_DIR=docs/showcase

:: Создаем папку, если её нет
if not exist %OUT_DIR% mkdir %OUT_DIR%

echo ========================================================
echo Generating Showcase Images for README.md
echo ========================================================
echo.

echo [1/5] Baseline: Auto mode, no TGC compensation...
python %SCRIPT% -i %INPUT% -o %OUT_DIR%/01_auto_tgc_off.png -s 900x1500 --auto --tgc-lift off

echo.
echo [2/5] Safe Mode: Global hard floor (protects all shadows)...
python %SCRIPT% -i %INPUT% -o %OUT_DIR%/02_auto_tgc_safe.png -s 900x1500 --auto --tgc-lift safe 1.0

echo.
echo [3/5] Stress Mode: Smart per-pixel floor (Recommended for Tarot/Cards)...
python %SCRIPT% -i %INPUT% -o %OUT_DIR%/03_auto_tgc_stress.png -s 900x1500 --auto --tgc-lift stress 1.0

echo.
echo [4/5] Advanced: Stress + Soft Floor (preserves gradients) + Prepress Risk Map...
python %SCRIPT% -i %INPUT% -o %OUT_DIR%/04_advanced_prepress.png -s 900x1500 --auto --tgc-lift stress 1.2 --soft-floor 2.5 --prepress-filter

echo.
echo [5/5] Pro-Tip: Black Anchor (preserves true Dmax/Deep blacks) + QA Sheet...
python %SCRIPT% -i %INPUT% -o %OUT_DIR%/05_black_anchor_qa.png -s 900x1500 --auto --tgc-lift stress 1.0 --black-anchor 0.03 --qa-dir %OUT_DIR%/qa

echo.
echo ========================================================
echo Done! All showcase images are in: %OUT_DIR%
echo ========================================================
pause