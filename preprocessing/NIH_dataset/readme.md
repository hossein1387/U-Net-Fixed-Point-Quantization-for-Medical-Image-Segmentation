If the mask and the image are misaligned like it should be after the download launch the realignement.sh script first. Change the path at the beginning of the file. It should point to where your data files are.  
Name the image folder 'data' and the mask folder 'label'. data and label contain nifti image. Change the path_glob value in the head of the .py files to the folder where data and label are. The original data can be downloaded from academic torrent (http://academictorrents.com/details/80ecfefcabede760cdbdf63e38986501f7becd49)
Run looping.sh which will yield among other img_roi and label_roi. These are the picture used to feed the network. 
We then moved roughly the last ten pourcent to another folder : val_img_roi and val_label_roi 
Perform training with img_roi and label_roi and testing with val_img_roi and val_label_roi
