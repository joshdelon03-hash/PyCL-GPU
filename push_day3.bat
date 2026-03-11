@echo off
cd /d "C:\Users\user1\Desktop\PyCL-GPU-main"
git add amp/pool.py framework/neural.py main_neural.py
git commit -m "Push Week Day 3: Neural-Driven Dynamic Scaling & Robustness - Integrated PyTorch Latency Predictor for intelligent worker orchestration and data-pressure management."
git push origin main
echo Day 3 Pushed at %DATE% %TIME% >> push_log.txt
