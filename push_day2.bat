@echo off
cd /d "C:\Users\user1\Desktop\PyCL-GPU-main"
git add framework/general.py framework/algo/dpj.py framework/algo/__init__.py main_dpj.py
git add amp/*.py setup_amp.py
git commit -m "Push Week Day 2: Generalized AMP & Dot-Product Join (Radix Reordering) - Bundling the updated AMP library with the framework for easy installation."
git push origin main
echo Day 2 Pushed at %DATE% %TIME% >> push_log.txt
