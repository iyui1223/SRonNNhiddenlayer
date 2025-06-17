#!/bin/bash
# relocate the force_message files to the viewer directory
# Once is done, the quick_viewer.html can be used to view the force_message files by running:
# python3 -m http.server 8000 
# at the Viewer directory and accessing the url http://localhost:8000/quick_viewer.html

set -e
source "../Const/const_template.txt" # load the shared constants among experiment types

source ~/venvs/final_clean/bin/activate


FIG_DIR="${ROOT_DIR}/Figs"
DEST_DIR="${ROOT_DIR}/Viewer/force_message"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

cd $FIG_DIR/
# Find matching files and create symbolic links with flattened names
find "." -name "*force_message*" | while read -r filepath; do
  # Flatten path: replace / with _ and strip leading Figs/
  rel_path="${filepath#$SRC_DIR/}"
  rel_path="${rel_path#./}"
  flat_name="${rel_path//\//_}"

  # Create the symbolic link
  cp "$(realpath "$filepath")" "$DEST_DIR/$flat_name"
  echo $flat_name
done

