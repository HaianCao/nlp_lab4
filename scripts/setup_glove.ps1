Write-Host "================================" -ForegroundColor Green
Write-Host "Installing Python dependencies..." -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

pip install gensim numpy scipy

Write-Host ""
Write-Host "================================" -ForegroundColor Yellow
Write-Host "Downloading GloVe model..." -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
Write-Host ""
$pythonScript = @"
import gensim.downloader as api
import sys

def main():
    try:
        print("Downloading GloVe Wiki Gigaword 50d model...")
        model = api.load("glove-wiki-gigaword-50")
        print("✅ Model downloaded successfully!")
        print(f"📊 Vocabulary size: {len(model):,}")
        
        if "king" in model and "queen" in model:
            similarity = model.similarity("king", "queen")
            if similarity <= 0:
                return 1
            
        print("✨ Model is ready to use!")
        return 0
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"@

$pythonScript | Out-File -FilePath "temp_download.py" -Encoding UTF8
python temp_download.py

Remove-Item "temp_download.py" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "You can now use the GloVe model in your Python scripts." -ForegroundColor Cyan
Write-Host ""

Write-Host "Setup completed automatically." -ForegroundColor Cyan