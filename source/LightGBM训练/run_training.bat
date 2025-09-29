@echo off
echo 启动LightGBM训练管道...
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 运行训练管道
python run_training_pipeline.py

echo.
echo 训练完成，按任意键退出...
pause
