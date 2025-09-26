@echo off
chcp 65001 >nul
echo ========================================
echo IC检测工作流批量版 - 测试模式
echo ========================================
echo.
echo 正在启动测试模式...
echo 股票数量: 100
echo 因子数量: 50
echo 批次大小: 500
echo.
pause
python run_batch.py --mode test
pause
