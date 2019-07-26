#!/bin/bash


python gene_image.py  
wait $!
''' data is the folder where the nifti are you should add _png to the name and write the path here'''
d=$(ls /Users/lurou_admin/Desktop/pancreas_test/data_png| wc -l )
a=$(($((2*$d))));
echo $a
sleep 15 
python generation.py & v=$!
c=$(ls /Users/lurou_admin/Desktop/pancreas_test/image_tmp | wc -l )
while [ $c -le $a ]
do
  c=$(ls /Users/lurou_admin/Desktop/pancreas_test/image_tmp | wc -l)
done 
echo 'done'
kill $v
echo "$c"

python transform_im.py 

