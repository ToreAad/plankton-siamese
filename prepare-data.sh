
# sort data into train/validate/test directories
export SET="train validate test"

export cwd=$(pwd)
export data=data/

prepare_dir(){
    export dir=$1
    echo $dir
    for s in $SET; do 
        mkdir -p "$cwd/$s/$dir"
    done
    
    export tmp=$(mktemp)
    ls "$data/$dir" | shuf > $tmp

    head -100 $tmp | while read f; do
        ln -s "$cwd/$data/$dir/$f" "$cwd/validate/$dir/$f"
    done
    tail -n +101 $tmp | head -100 | while read f; do
        ln -s "$cwd/$data/$dir/$f" "$cwd/test/$dir/$f"
    done
    tail -n +201 $tmp | while read f; do
        ln -s "$cwd/$data/$dir/$f" "$cwd/train/$dir/$f"
    done

    rm $tmp
}
export -f prepare_dir

parallel -u --eta prepare_dir ::: $(ls -1 $data)


