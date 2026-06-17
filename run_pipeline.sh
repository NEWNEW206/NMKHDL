#!/bin/bash
echo "====================================================="
echo "  CHAY CHUONG TRINH DO AN NHAP MON KHOA HOC DU LIEU"
echo "===================================================== "

echo "[1] Dang kiem tra va cai dat thu vien tu requirements.txt..."
pip install -r requirements.txt

echo ""
echo "[2] Dang tien hanh chay Pipeline Machine Learning..."
export PYTHONIOENCODING=utf-8
python main.py

echo ""
echo "==================================================="
echo "  HOAN TAT! Vui long kiem tra thu muc 'outputs/'"
echo "==================================================="
