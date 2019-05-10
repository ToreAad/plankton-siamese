
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
    echo "Converting " "$CWD/$BASE/$DIR/$f" "$CWD/$OUT/$DIR/$f"
    convert -resize 299x299 "$CWD/$BASE/$DIR/$f" -background white -gravity center -extent 299x299 "$CWD/$OUT/$DIR/$f"
}
export -f convert_file

convert_dir(){
    export DIR=$1
    mkdir -p "$OUT/$DIR"
    echo "Converting from " "$DIR"
    parallel convert_file ::: $(ls -1 "$CWD/$BASE/$1" | head)
}
export -f convert_dir
echo "Starting parallel convert_dir"
parallel convert_dir ::: $(ls -1 $BASE | head)
