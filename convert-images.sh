
# safety first
set -e -o pipefail
shopt -s failglob 

# resize to 299x299, but keep image size
# drop -resize if zooming with crop is better (but maybe not?)

BASE=./imgs
OUT=./data
CWD=$(pwd)

convert_file(){
    f = $1
    echo "$CWD/$BASE/$DIR/$f" "$CWD/$OUT/$DIR/$f"
}

convert_dir(){
    DIR = $1
    #echo "$DIR"
    #mkdir -p "$OUT/$DIR"

    parallel convert_file ::: $(ls -1 "$BASE/$DIR")

    # ls "$BASE/$DIR" | while read f; do 
    #     echo "$CWD/$BASE/$DIR/$f" "$CWD/$OUT/$DIR/$f"
        #echo "$f"
        # convert -resize 299x299 "$CWD/$BASE/$DIR/$f" -background white -gravity center -extent 299x299 "$CWD/$OUT/$DIR/$f"
    # done
}
parallel convert_dir ::: $(ls -1 $BASE)