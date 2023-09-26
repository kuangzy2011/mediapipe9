#!/bin/bash

if [ $# -eq 1 ]; then
    old_path=$(pwd)
    cd $1
    ls -l
    echo "{" > label.txt
    idx=0
    for n in $(ls $1); do
        if [ $n != "label.txt" ]; then
            echo "    ${idx}: \"$n\"," >> label.txt
            idx=`expr $idx + 1`
        fi
    done
    echo "}" >> label.txt
    #cd ..; zip -r handata.zip $(basename $1);mv handata.zip $old_path
    cd $old_path
fi
