@echo off
chcp 65001
echo ========================================
echo IC检测工作流启动脚本
echo ========================================

cd /d "%~dp0"

echo 当前目录: %CD%
echo.

echo 开始执行IC检测工作流...
python run_ic_workflow.py

echo.
echo 工作流执行完成！
pause
