# Railway-Safety

## data

### mapillary

`image_seq/`
- Street-level images
- `df_image_seq.csv`
    - Metadata table for `image_seq/`
    - Columns:
        - `crossing_id`: Crossing ID.
        - `seq_id`: ID for each group of image sequences (matches folder names).
        - `img_pos`: Position index of the image within a sequence.
        - `img_id`: Image ID (matches filename suffix).
        - `bearing`: Compass direction of the camera facing toward the center of the crossing.
- *[92 image sequence folders]*
    - Named with 22-digit sequence ID (e.g. `0Ct3ZUQwHMklErx1YDm4fA/`)
    - *[a sequence of images]*
        - Named with `[img_pos]_[img_id].jpg` (e.g. `0834_1401964997887726.jpg`)
        - Within a range of 10~20m from a railgrade crossing
