@echo off
echo ================================
echo Installing Python dependencies...
echo ================================

pip install gensim numpy scipy

echo.
echo ================================
echo Downloading GloVe model...
echo ================================
echo This may take a few minutes...
echo.
echo import gensim.downloader as api > temp_download.py
echo import sys >> temp_download.py
echo. >> temp_download.py
echo def main(): >> temp_download.py
echo     try: >> temp_download.py
echo         print("Downloading GloVe Wiki Gigaword 50d model...") >> temp_download.py
echo         model = api.load("glove-wiki-gigaword-50") >> temp_download.py
echo         print("✅ Model downloaded successfully!") >> temp_download.py
echo         print("📊 Vocabulary size: {len(model):,}") >> temp_download.py
echo. >> temp_download.py
echo         if "king" in model and "queen" in model: >> temp_download.py
echo             similarity = model.similarity("king", "queen") >> temp_download.py
echo             if similarity ^<= 0: >> temp_download.py
echo                 return 1 >> temp_download.py
echo. >> temp_download.py
echo         print("✨ Model is ready to use!") >> temp_download.py
echo         return 0 >> temp_download.py
echo. >> temp_download.py
echo     except Exception as e: >> temp_download.py
echo         print(f"❌ Error downloading model: {e}") >> temp_download.py
echo         return 1 >> temp_download.py
echo. >> temp_download.py
echo if __name__ == "__main__": >> temp_download.py
echo     sys.exit(main()) >> temp_download.py

python temp_download.py

del temp_download.py

echo.
echo ================================
echo Setup complete!
echo ================================
echo You can now use the GloVe model in your Python scripts.
echo.
echo Setup completed automatically.