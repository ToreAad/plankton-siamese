
# safety first
set -e -o pipefail
shopt -s failglob 

# resize to 299x299, but keep image size
# drop -resize if zooming with crop is better (but maybe not?)

export BASE=ZooScanSet/imgs
export OUT=data
export CWD=$(pwd)

convert_file(){
    f=$1
    echo "$CWD/$BASE/$DIR/$f" "$CWD/$OUT/$DIR/$f"
}
export convert_file

convert_dir(){
    export DIR=$1
    #echo "$DIR"
    #mkdir -p "$OUT/$DIR"
    echo "Starting parallel convert_file"
    parallel convert_file ::: $(ls -1 "$CWD/$BASE/$1" | head)

    # ls "$BASE/$DIR" | while read f; do 
    #     echo "$CWD/$BASE/$DIR/$f" "$CWD/$OUT/$DIR/$f"
        #echo "$f"
        # convert -resize 299x299 "$CWD/$BASE/$DIR/$f" -background white -gravity center -extent 299x299 "$CWD/$OUT/$DIR/$f"
    # done
}
export convert_dir
echo "Starting parallel convert_dir"
parallel convert_dir ::: $(ls -1 $BASE | head)
