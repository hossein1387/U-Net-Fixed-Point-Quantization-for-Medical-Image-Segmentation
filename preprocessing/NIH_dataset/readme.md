If the mask and the image are misaligned like it should be after the download launch the realignement script first. 
Name the image folder 'data' and the mask folder 'label'. data and label contain nifti image. Change the path_glob value in the head of the .py files to the folder where data and label are. The original data can be downloaded from academic torrent (http://academictorrents.com/details/80ecfefcabede760cdbdf63e38986501f7becd49)
then run looping.sh which will yield among other img_roi and label_roi. These are the picture used in the study. 
We then moved roughly the last ten pourcent to another folder : val_img_roi and val_label_roi 
so train with img roi and label roi , validation with val_img_roi and val_label_roi
