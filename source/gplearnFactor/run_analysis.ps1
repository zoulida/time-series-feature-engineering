# PowerShell脚本：运行GPLearn时序特征提取和IC测试
# 设置控制台编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "GPLearn时序特征提取和IC测试程序" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 步骤1: 检查Python环境
Write-Host "步骤1: 检查Python环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 错误: Python未安装或不在PATH中" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

Write-Host ""
# 步骤2: 测试gplearn是否可用
Write-Host "步骤2: 测试gplearn是否可用..." -ForegroundColor Yellow
$testResult = python test_gplearn.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "gplearn测试失败，正在安装..." -ForegroundColor Yellow
    python install_gplearn.py
    
    Write-Host ""
    Write-Host "重新测试gplearn..." -ForegroundColor Yellow
    python test_gplearn.py
}

Write-Host ""
# 步骤3: 运行特征提取分析
Write-Host "步骤3: 运行特征提取分析..." -ForegroundColor Yellow
python run_gplearn_analysis.py

Write-Host ""
Write-Host "程序执行完成！" -ForegroundColor Green
Read-Host "按回车键退出"
