@echo off
chcp 65001 >nul
echo ============================================================
echo GPLearn时序特征提取和IC测试程序
echo ============================================================
echo.

echo 步骤1: 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo 步骤2: 测试gplearn是否可用...
python test_gplearn.py
if errorlevel 1 (
    echo.
    echo gplearn测试失败，正在安装...
    python install_gplearn.py
    echo.
    echo 重新测试gplearn...
    python test_gplearn.py
)

echo.
echo 步骤3: 运行特征提取分析...
python run_gplearn_analysis.py

echo.
echo 程序执行完成！
pause
