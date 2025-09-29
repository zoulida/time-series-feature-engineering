# PowerShell脚本 - 运行LightGBM训练管道
Write-Host "启动LightGBM训练管道..." -ForegroundColor Green
Write-Host ""

# 检查Python是否可用
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Yellow
} catch {
    Write-Host "错误: Python未安装或不在PATH中" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 运行训练管道
Write-Host "开始执行训练管道..." -ForegroundColor Cyan
python run_training_pipeline.py

Write-Host ""
Write-Host "训练完成!" -ForegroundColor Green
Read-Host "按回车键退出"
