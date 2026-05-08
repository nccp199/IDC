# 只提交指定文件，避免误上传模型、CSV、图片、输出文件夹

$files = @(
    "config_ultimate.py",
    "eval_nn_reuse.py",
    "git_push_selected.ps1"
    "IDCPriceEnv20D_ultimate.py"
    "train_ppo_ultimate.py"
    "ga_base.py"
    "pso_base.py"
    "eval_base.py"
    "task_model.py"
    "power_model.py"
    "task.py"
    "demo_random_env_test.py"
    "demo_task_model.py"

)

Write-Host "=== Git status before add ==="
git status

Write-Host "`n=== Clear staged files ==="
git restore --staged .

Write-Host "`n=== Add selected files ==="
foreach ($file in $files) {
    if (Test-Path $file) {
        git add $file
        Write-Host "Added: $file"
    } else {
        Write-Host "Skip missing file: $file"
    }
}

Write-Host "`n=== Git status after add ==="
git status

$time = Get-Date -Format "yyyy-MM-dd HH:mm"
$msg = "update selected files $time"

Write-Host "`n=== Commit ==="
git commit -m "$msg"

Write-Host "`n=== Push ==="
git push

Write-Host "`nDone."