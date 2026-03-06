#!/bin/bash
# =============================================================
# Run this script on the SERVER after experiments complete.
# It extracts metrics, prepares key files, and syncs via git.
#
# Usage:
#   bash scripts/server_sync.sh [results_dir1] [results_dir2] ...
#   bash scripts/server_sync.sh paper_results paper_results_s2
# =============================================================

set -e

RESULTS_DIRS="${@:-paper_results}"
echo "=============================================="
echo "Server Sync: Preparing results for git"
echo "=============================================="
echo "Results dirs: $RESULTS_DIRS"

# Step 1: Show sizes
echo ""
echo "--- Directory sizes ---"
for d in $RESULTS_DIRS; do
    if [ -d "$d" ]; then
        echo "  $d: $(du -sh $d | cut -f1)"
    else
        echo "  $d: NOT FOUND"
    fi
done

# Step 2: Run metrics extraction
echo ""
echo "--- Extracting metrics ---"
python scripts/sync_results.py $RESULTS_DIRS

# Step 3: Git add only key files
echo ""
echo "--- Staging key files for git ---"

# Source code
git add .gitignore scripts/ *.py configs/ experiments/ utils/ data/ 2>/dev/null || true
git add requirements.txt environment.yaml 2>/dev/null || true

# Merged results
git add merged_results.json 2>/dev/null || true

# Per-directory key files
for d in $RESULTS_DIRS; do
    if [ -d "$d" ]; then
        git add "$d/all_results.json" 2>/dev/null || true
        git add "$d/checkpoint.json" 2>/dev/null || true
        git add "$d/latex/" 2>/dev/null || true
        # Only add qualitative/attention figures (small, needed for paper)
        git add "$d/figures/qualitative/" 2>/dev/null || true
        git add "$d/figures/attention/" 2>/dev/null || true
        git add "$d/figures/curvature_plot.pdf" 2>/dev/null || true
        git add "$d/figures/curvature_plot.py" 2>/dev/null || true
    fi
done

# Step 4: Show what will be committed
echo ""
echo "--- Files staged for commit ---"
git diff --cached --stat
echo ""
STAGED_SIZE=$(git diff --cached --stat | tail -1)
echo "Summary: $STAGED_SIZE"

echo ""
echo "--- Ready to commit. Run: ---"
echo '  git commit -m "Sync experiment results"'
echo '  git push'

