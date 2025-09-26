@echo off
chcp 65001 >nul
echo ========================================
echo IC检测工作流批量版 - 生产模式
echo ========================================
echo.
echo 正在启动生产模式...
echo 股票数量: 3500
echo 因子数量: 242
echo 批次大小: 500
echo.
echo 警告: 生产模式将运行较长时间（2-3小时）
echo 请确保有足够的内存和磁盘空间
echo.
pause
python run_batch.py --mode production
pause
