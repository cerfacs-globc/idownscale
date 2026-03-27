#!/bin/bash
# setup_workspace.sh
# Automatically adapts iriscc/settings.py absolute paths to the current workspace.

NEW_ROOT=$(pwd)
SETTINGS_FILE="iriscc/settings.py"

if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "Error: Must be run from the repository root."
    exit 1
fi

echo "Adapting $SETTINGS_FILE to $NEW_ROOT..."

# Use a safe delimiter | for sed because paths contain /
sed -i "s|/scratch/globc/page/idownscale_active|$NEW_ROOT|g" "$SETTINGS_FILE"

echo "Done. All project paths now point to $NEW_ROOT."
echo "You can now run the pipeline: sbatch run_exp5_full.sh"
