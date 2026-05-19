# Start FastAPI Server
Write-Host "=== Starting FastAPI Server ===" -ForegroundColor Green
Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1" -NoNewWindow
Start-Sleep -Seconds 3

# Start Celery Worker
Write-Host "=== Starting Celery Worker ===" -ForegroundColor Green
Start-Process -FilePath "python" -ArgumentList "-m", "celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info" -NoNewWindow
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== All services started! ===" -ForegroundColor Green
Write-Host "API Server: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop services" -ForegroundColor Yellow
