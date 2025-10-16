# PowerShell script to cleanup GloVe models
Write-Host "================================" -ForegroundColor Yellow
Write-Host "🗑️  GloVe Model Cleanup Script" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

# Check if Python and gensim are available
Write-Host "📍 Checking gensim installation..." -ForegroundColor Blue

try {
    $gensimDir = python -c "import gensim.downloader as api; print(api.base_dir)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: Cannot find gensim installation" -ForegroundColor Red
        Write-Host "Make sure gensim is installed: pip install gensim" -ForegroundColor Cyan
        exit 1
    }
} catch {
    Write-Host "❌ Error: Python or gensim not found" -ForegroundColor Red
    exit 1
}

Write-Host "📂 Gensim cache directory: $gensimDir" -ForegroundColor Cyan

# Check if directory exists
if (!(Test-Path $gensimDir)) {
    Write-Host "✅ No gensim cache found - already clean!" -ForegroundColor Green
    exit 0
}

# List what's in the cache
Write-Host ""
Write-Host "📋 Current cached models:" -ForegroundColor Blue
try {
    $items = Get-ChildItem $gensimDir -ErrorAction SilentlyContinue
    if ($items) {
        foreach ($item in $items) {
            if ($item.PSIsContainer) {
                Write-Host "   📁 $($item.Name)/" -ForegroundColor White
            } else {
                Write-Host "   📄 $($item.Name)" -ForegroundColor White
            }
        }
    } else {
        Write-Host "   📭 Cache directory is empty" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ Error reading cache directory" -ForegroundColor Red
}

Write-Host ""
Write-Host "⚠️  This will delete ALL cached gensim models!" -ForegroundColor Yellow
$response = Read-Host "Do you want to continue? (y/N)"

if ($response -match '^[Yy]([Ee][Ss])?$') {
    Write-Host ""
    Write-Host "🧹 Cleaning up gensim cache..." -ForegroundColor Blue
    
    try {
        # Remove the entire gensim-data directory
        Remove-Item $gensimDir -Recurse -Force -ErrorAction Stop
        
        Write-Host "✅ Successfully deleted gensim cache!" -ForegroundColor Green
        Write-Host "💾 Freed up disk space" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "📊 Cache status:" -ForegroundColor Cyan
        if (!(Test-Path $gensimDir)) {
            Write-Host "   • Cache directory: DELETED ✅" -ForegroundColor Green
        } else {
            Write-Host "   • Cache directory: STILL EXISTS ❌" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Error: Failed to delete cache" -ForegroundColor Red
        Write-Host "Try running as Administrator or check permissions" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "👍 Operation cancelled - no changes made" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🎉 Cleanup script completed!" -ForegroundColor Green