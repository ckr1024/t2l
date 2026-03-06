#!/bin/bash
# =============================================================
# Run this ONCE on the server to set up .gitignore and scripts
# WITHOUT using git pull (avoids any risk to experiment data).
#
# Usage:
#   bash scripts/setup_server.sh
#
# What it does:
#   1. Updates .gitignore to exclude bulk images
#   2. Untracks already-committed bulk files (keeps them on disk)
#   3. Installs the sync_results.py script
# =============================================================

set -e
echo "=============================================="
echo "Server Setup: Configure git for results sync"
echo "=============================================="

# Step 1: Update .gitignore
echo ""
echo "--- Step 1: Updating .gitignore ---"
cat > .gitignore << 'GITIGNORE_EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model cache
*.safetensors
*.bin
*.ckpt
*.pt
*.pth

# ============================================================
# Experiment results: ONLY sync metrics, tables, and key figures
# ============================================================

# Ignore ALL generated images in experiment result directories
paper_results*/*/images/
paper_results*/**/images/
paper_results*/time/*/

# Ignore mask images (generated during pipeline)
mask_*

# Keep these (not ignored):
#   paper_results*/all_results.json
#   paper_results*/checkpoint.json
#   paper_results*/latex/*.tex
#   paper_results*/figures/curvature_plot.pdf
#   paper_results*/figures/curvature_plot.py
#   paper_results*/figures/qualitative/*.png  (few, for paper)
#   paper_results*/figures/attention/*.png    (few, for paper)

# Ignore any large binary outputs
*.mp4
*.avi
*.zip
*.tar.gz
GITIGNORE_EOF
echo "  .gitignore updated"

# Step 2: Untrack bulk files from git index (keeps files on disk!)
echo ""
echo "--- Step 2: Untracking bulk files (files stay on disk) ---"

# Remove pyc from tracking
TRACKED_PYC=$(git ls-files | grep '\.pyc$' 2>/dev/null | wc -l)
if [ "$TRACKED_PYC" -gt 0 ]; then
    git ls-files | grep '\.pyc$' | xargs git rm --cached 2>/dev/null || true
    echo "  Untracked $TRACKED_PYC .pyc files"
fi

# Remove .idea from tracking
if git ls-files | grep -q '\.idea/'; then
    git rm -r --cached .idea/ 2>/dev/null || true
    echo "  Untracked .idea/"
fi

# Remove bulk images from tracking
TRACKED_IMGS=$(git ls-files | grep 'paper_results.*images.*\.png$' 2>/dev/null | wc -l)
if [ "$TRACKED_IMGS" -gt 0 ]; then
    git ls-files | grep 'paper_results.*images.*\.png$' | xargs git rm --cached 2>/dev/null || true
    echo "  Untracked $TRACKED_IMGS bulk image files"
fi

TRACKED_TIME=$(git ls-files | grep 'paper_results.*/time/' 2>/dev/null | wc -l)
if [ "$TRACKED_TIME" -gt 0 ]; then
    git ls-files | grep 'paper_results.*/time/' | xargs git rm --cached 2>/dev/null || true
    echo "  Untracked $TRACKED_TIME time experiment files"
fi

echo ""
echo "--- Step 3: Verify ---"
echo "Tracked files remaining:"
git ls-files | wc -l
echo ""
echo "Tracked PNG files (should be only qualitative/demo):"
git ls-files | grep '\.png$' | wc -l
echo ""

echo "=============================================="
echo "Setup complete! Now you can run:"
echo "  python scripts/sync_results.py paper_results paper_results_s2"
echo "  bash scripts/server_sync.sh paper_results paper_results_s2"
echo "=============================================="

