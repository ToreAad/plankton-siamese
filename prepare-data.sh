
# sort data into train/validate/test directories
SET="train validate test"

tmp=$(mktemp)
cwd=$(pwd)

while read dir; do 
    echo $dir
    for s in $SET; do 
        mkdir -p "$s/$dir"
    done
    ls "data/$dir" | shuf > $tmp
    # test images
    head -100 $tmp | while read f; do
        ln -s "$cwd/data/$dir/$f" "validate/$dir/$f"
    done
    tail -n +101 $tmp | head -100 | while read f; do
        ln -s "$cwd/data/$dir/$f" "test/$dir/$f"
    done
    tail -n +201 $tmp | while read f; do
        ln -s "$cwd/data/$dir/$f" "train/$dir/$f"
    done
    rm $tmp
done

