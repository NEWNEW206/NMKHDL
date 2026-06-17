@echo off
echo =====================================================
echo   CHAY MINI PROJECT NHAP MON KHOA HOC DU LIEU
echo =====================================================

echo [1] Dang kiem tra va cai dat thu vien tu requirements.txt...
pip install -r requirements.txt

echo.
echo [2] Dang tien hanh chay Pipeline Machine Learning...
set PYTHONIOENCODING=utf-8

python main.py

echo.
echo =========================================================
echo   CHAY THANH CONG! Vui long kiem tra thu muc 'outputs/'
echo =========================================================
pause