# Autonomous Navigation - Full Windows Benchmark & Evaluation Pipeline
# Usage: .\scripts\run_pipeline_windows.ps1 [-Episodes 100] [-Seed 0]

param (
    [int]$Episodes = 100,
    [int]$Seed = 0,
    [string]$Model = "models/ppo_navigation"
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Autonomous Navigation - Full Research Evaluation Pipeline" -ForegroundColor Yellow
Write-Host "   Episodes: $Episodes | Seed: $Seed | Model: $Model" -ForegroundColor Gray
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Environment Verification
Write-Host "[1/6] Validating environment..." -ForegroundColor Cyan
python -m rl_training.check_env
if ($LASTEXITCODE -ne 0) { Exit 1 }

# 2. Evaluate PPO Policy
Write-Host "[2/6] Evaluating trained PPO policy on unseen test scenarios..." -ForegroundColor Cyan
python -m rl_training.evaluate --model $Model --episodes $Episodes --split test --seed $Seed --out results/ppo_evaluation.csv

# 3. Evaluate Classical Baselines (Reactive & DWA)
Write-Host "[3/6] Evaluating classical baselines (Reactive & DWA)..." -ForegroundColor Cyan
python -m rl_training.baselines --episodes $Episodes --split test --seed $Seed --out results/baselines_evaluation.csv

# 4. Multi-Condition Matrix Comparison (Nominal, Lidar Noise, Loc Drift across Test & Heldout)
Write-Host "[4/6] Running full factorial controller comparison (E1 - E6a)..." -ForegroundColor Cyan
python -m rl_training.compare --model $Model --episodes $Episodes --seed $Seed

# 5. Generate Qualitative Trajectory Figures
Write-Host "[5/6] Generating trajectory plots for report..." -ForegroundColor Cyan
python -m rl_training.visualize --model $Model --split test --n 6 --out results/figures/trajectories_test.png
python -m rl_training.visualize --model $Model --split heldout --n 6 --out results/figures/trajectories_heldout.png

# 6. Generate Academic Charts and LaTeX Tables
Write-Host "[6/6] Generating publication charts and academic tables..." -ForegroundColor Cyan
python -m analysis.generate_visual_comparison
python -m analysis.export_academic_tables

Write-Host "=================================================================" -ForegroundColor Green
Write-Host " Pipeline execution complete! Check outputs in results/:" -ForegroundColor Yellow
Write-Host "   Summary table : results\compare_summary.md" -ForegroundColor White
Write-Host "   LaTeX tables  : results\table_comparison.tex" -ForegroundColor White
Write-Host "   Figures       : results\figures\" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Green
