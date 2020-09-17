#!/bin/bash

f=$1

echo "Shrinking figure "$f
gs -dBATCH -dNOPAUSE -dSAFER -dQUIET -dNOPLATFONTS -dSubsetFonts=true -sDEVICE=pdfwrite -sOutputFile=tmp.pdf $f
sleep 3
oldsize=$(du -k $f | cut -f 1)
newsize=$(du -k tmp.pdf | cut -f 1)
echo "Old size is : "$oldsize" New size is : "$newsize

gs -dBATCH -dNOPAUSE -dSAFER -dQUIET -dNOPLATFONTS -dSubsetFonts=true -sDEVICE=pdfwrite -sOutputFile=tmp2.pdf tmp.pdf
sleep 3
newsize2=$(du -k tmp2.pdf | cut -f 1)
echo "Second size is :"$newsize2

if [ $newsize -le $oldsize ]; then
    echo "Moving file "$f
    mv tmp.pdf $f
    oldsize=$newsize
else
    rm -f tmp.pdf
    echo "File "$f" retained"
fi

if [ $newsize2 -le $oldsize ]; then
    echo "Moving file "$f" on second iteration"
    mv tmp2.pdf $f
else
    rm -f tmp2.pdf
    echo "File "$f" retained on second pass"
fi
