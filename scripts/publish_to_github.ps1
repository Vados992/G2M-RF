param(
  [string]$Repo = "https://github.com/Vados992/G2M-RF.git"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".git")) { git init -b main }
git add .
if (-not (git status --porcelain)) { Write-Host "No changes to commit." }
else { git commit -m "Initial G2M-RF v2.0 executable research framework" }
$remotes = git remote
if ($remotes -contains "origin") { git remote set-url origin $Repo }
else { git remote add origin $Repo }
git push -u origin main
