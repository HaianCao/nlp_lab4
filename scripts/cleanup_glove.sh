#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}🗑️  GloVe Model Cleanup Script${NC}"
echo -e "${YELLOW}================================${NC}"

# Get gensim cache directory
echo -e "${BLUE}📍 Checking gensim cache location...${NC}"

GENSIM_DIR=$(python -c "import gensim.downloader as api; print(api.base_dir)" 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: Cannot find gensim installation${NC}"
    echo -e "${CYAN}Make sure gensim is installed: pip install gensim${NC}"
    exit 1
fi

echo -e "${CYAN}📂 Gensim cache directory: ${GENSIM_DIR}${NC}"

# Check if directory exists
if [ ! -d "$GENSIM_DIR" ]; then
    echo -e "${GREEN}✅ No gensim cache found - already clean!${NC}"
    exit 0
fi

# List what's in the cache
echo ""
echo -e "${BLUE}📋 Current cached models:${NC}"
ls -la "$GENSIM_DIR" 2>/dev/null || echo -e "${YELLOW}📭 Cache directory is empty${NC}"

echo ""
echo -e "${YELLOW}⚠️  This will delete ALL cached gensim models!${NC}"
echo -e "${CYAN}Do you want to continue? (y/N):${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo -e "${BLUE}🧹 Cleaning up gensim cache...${NC}"
    
    # Remove the entire gensim-data directory
    rm -rf "$GENSIM_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully deleted gensim cache!${NC}"
        echo -e "${GREEN}💾 Freed up disk space${NC}"
        echo ""
        echo -e "${CYAN}📊 Cache status:${NC}"
        if [ ! -d "$GENSIM_DIR" ]; then
            echo -e "${GREEN}   • Cache directory: DELETED${NC}"
        else
            echo -e "${RED}   • Cache directory: STILL EXISTS${NC}"
        fi
    else
        echo -e "${RED}❌ Error: Failed to delete cache${NC}"
        echo -e "${YELLOW}Try running with sudo or check permissions${NC}"
        exit 1
    fi
else
    echo -e "${CYAN}👍 Operation cancelled - no changes made${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Cleanup script completed!${NC}"
echo ""
echo -e "${CYAN}Cleanup completed automatically.${NC}"