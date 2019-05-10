
# sort data into train/validate/test directories
SET="train validate test"

tmp=$(mktemp)
cwd=$(pwd)
data=data/

prepare_file(){
    # BASE=$1
    # f=$2
    # ln -s "$cwd/$data/$dir/$f" "$BASE/$dir/$f"
    echo "$cwd/$data/$dir/$f" "$BASE/$dir/$f"
}
export -f prepare_file

prepare_dir(){
    export dir=$1
    echo $dir
    # for s in $SET; do 
    #     mkdir -p "$s/$dir"
    # done
    # ls "$data/$dir" | shuf > $tmp

    parallel echo "validate" {} \| prepare_file ::: $(head -100 $tmp)
    parallel echo "test" {} \| prepare_file ::: $(tail -n +101 $tmp | head -100)
    parallel echo "train" {} \| prepare_file ::: $(tail -n +201 $tmp)

    # rm $tmp
}
export -f prepare_dir

 parallel prepare_dir ::: $(ls -1 $data)


