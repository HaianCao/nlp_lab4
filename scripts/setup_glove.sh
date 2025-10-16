#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Installing Python dependencies...${NC}"
echo -e "${GREEN}================================${NC}"

if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${BLUE}Installing gensim, numpy, scipy...${NC}"
pip install gensim numpy scipy

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to install Python packages${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}Downloading GloVe model...${NC}"
echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"
echo ""
cat > temp_download.py << 'EOF'
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
EOF

python temp_download.py
download_status=$?

rm -f temp_download.py

echo ""
if [ $download_status -eq 0 ]; then
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}🎉 Setup complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo -e "${CYAN}You can now use the GloVe model in your Python scripts.${NC}"
    echo ""
    echo -e "${BLUE}Quick usage example:${NC}"
    echo -e "${CYAN}import gensim.downloader as api${NC}"
    echo -e "${CYAN}model = api.load('glove-wiki-gigaword-50')${NC}"
    echo -e "${CYAN}similar = model.most_similar('king', topn=5)${NC}"
else
    echo -e "${RED}================================${NC}"
    echo -e "${RED}❌ Setup failed!${NC}"
    echo -e "${RED}================================${NC}"
    echo -e "${YELLOW}Please check the error messages above.${NC}"
fi

echo ""
echo -e "${CYAN}Setup completed automatically.${NC}"