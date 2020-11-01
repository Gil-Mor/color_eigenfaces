# color eigenfaces
Are you tired of all these boring greyscale [eigenfaces](https://en.wikipedia.org/wiki/Eigenface)???

![greyscale eigenfaces](https://upload.wikimedia.org/wikipedia/commons/6/67/Eigenfaces.png)

## Create your own colored eigenfaces album.


Opencv is used to crop faces from all images in a directory. Then, [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is applied to get the eigenfaces.  

### The catch  

PCA can only work on 1 RGB channel at a time, so I split the images and apply PCA on each channel individually. Then I merge the outputs and the results are colored eigenfaces.

![colored eigenfaces](https://github.com/Gil-Mor/color_eigenfaces/blob/master/eigen_faces_color.png)

# Usage:
You can drop an image folder on the dragndrop.bat script.

Or from the cmd line:
python color_eigenfaces.py <path to images folder (can be relative to script)> [optional 'grey' for grey scale output]

Or you can manually run the code.

# Note:
opencv doesn't recognize faces at 100% so the scripts will ask you to go to the cropped_faces folder that was created 
inside your input imgs folder and delete all non-faces crops before it continues to get the eigenfaces.

# Requires:
python3, opencv, scipy, sklearn, matplotlib, imageio.

```bash
sudo apt update && \
sudo apt install -y python3-opencv && \
pip3 install matplotlib scipy imageio sklearn
```
