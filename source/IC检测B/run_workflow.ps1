# IC检测工作流PowerShell启动脚本
Write-Host "========================================" -ForegroundColor Green
Write-Host "IC检测工作流启动脚本" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 设置工作目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "当前目录: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

Write-Host "开始执行IC检测工作流..." -ForegroundColor Cyan
python run_ic_workflow.py

Write-Host ""
Write-Host "工作流执行完成！" -ForegroundColor Green
Read-Host "按任意键继续"
