@echo off
echo --- Day 4: A* PCIe Data Traversal Push ---
python -m pip install -r requirements.txt
python main_astar.py
echo Day 4 Logic Verified.
echo Results saved to astar_routing_result.png
echo.
echo Pushing to GitHub...
git add .
git commit -m "Day 4: A* PCIe Data Routing (FP16/FP32 Hardware Dispatch)"
git push origin main
echo Push Complete.
