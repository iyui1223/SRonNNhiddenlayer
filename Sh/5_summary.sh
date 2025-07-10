#!/bin/bash
# relocate the image files to the viewer directory

set -e
source "../Const/const_template.txt" # load the shared constants among experiment types

source ${SOURCE_ENV}


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

