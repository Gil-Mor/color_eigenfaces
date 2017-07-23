# eigenfaces_in_color
Are You Tired of all these boring grey scale eigenfaces???

Create your own colored eigenfaces album.
I use opencv to crop faces from all images in a directory and then apply PCA
to get the eigenfaces.
Unfortunately, PCA can only work on 1 RGB channel at a time, so I split the images
to channels and apply PCA on each channel and then I merge the eigenfaces.
The results are cool colored eigen faces.

Try it. Usage:
from cmdline:
python eigen_face.py <path to images folder (can be relative to scripts)> [optional 'grey' for grey scale output]

Or you can manually edit the code.

Requires python 3.5, opencv3, numpy, matplotlib, and scipy
I use Anaconda (As of 23.7.17 opencv module is not yet available for python 3.6 via pip or conda)
With python 3.5 You can install opencv3 with 'conda install -c menpo opencv3'
