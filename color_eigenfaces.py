
import cv2
import sys
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from sklearn.decomposition import PCA
import math

# resize cropped faces to this size - pca needs all images to be in the same size
faces_h = faces_w = 500

def plot_eigenfaces(images, h, w, color=False):
    """
    Plot eigen faces in matplotlib.
    :param images: list of images of shape (len(images), h*w)
    :param h: reshape images to (h,w)
    :param w: reshape images to (h,w)
    :param color: plot in color or grey scale
    """
    n_row = n_col = int(math.sqrt(len(images)))
    if color:
        plot_shape = (h, w, 3)
    else:
        plot_shape = (h,w)

    fig = plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))

    # set figure background to white instead of ugly grey
    fig.patch.set_facecolor('white')

    for i in range(n_row * n_col):
        ax = plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(plot_shape), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
        # color subplots borders in gold
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor('#e3dd00')
    plt.tight_layout()

# ------------------------------------------------------------------

def get_one_channel_face(im, rgb_index):
    """
    Get 1 RGB channel of an image.
    Given an image of shape (h,w,3) return a matrix of shape (h,w)
    """
    rgb = cv2.split(im) # at his point opencv doesn't transform the image to BGR.
    return rgb[rgb_index]
# ------------------------------------------------------------------

def get_all_faces_one_channel(faces, rgb_index):
    """
    Split all faces images to discrete r,g,b channels.
    """
    color_faces = []
    for face in faces:
        color_faces.append(get_one_channel_face(face, rgb_index))
    return np.array(color_faces)
# ------------------------------------------------------------------

def combine_color_channels(discrete_rgb_images):
    """
    Combine discrete r,g,b images to RGB iamges.
    :param discrete_rgb_images:
    :return:
    """
    color_imgs = []
    for r,g,b in zip(*discrete_rgb_images):
        r = (255 * (r - np.max(r)) / -np.ptp(r)).astype(int)
        g = (255 * (g - np.max(g)) / -np.ptp(g)).astype(int)
        b = (255 * (b - np.max(b)) / -np.ptp(b)).astype(int)

        color_imgs.append(cv2.merge((r, g, b)))
    return color_imgs

# ------------------------------------------------------------------

def get_color_eigen_faces(faces, n_components):
    """
    Splits the faces to single RGB channels, perform pca on each channel
    and in the end merges the eigenfaces to create color eigenfaces.
    return: array of colored eigenfaces.
    """

    # since faces were now opened with scipy and not opencv they are in
    # RGB format and not in BGR format.
    red_faces = get_all_faces_one_channel(faces, 0)
    green_faces = get_all_faces_one_channel(faces, 1)
    blue_faces = get_all_faces_one_channel(faces, 2)

    color_faces = [red_faces, green_faces, blue_faces]

    reshaped_color_faces = []
    for i, face in enumerate(color_faces):
        reshaped_color_faces.append(face.reshape((face.shape[0], face.shape[1] * face.shape[2])))

    color_faces = reshaped_color_faces
    color_pca_faces = []
    for color_face in color_faces:
        color_pca_faces.append(PCA(n_components=n_components, whiten=False).fit(color_face))

    color_eigen_faces = []
    for color_pca_face in color_pca_faces:
        color_eigen_faces.append(color_pca_face.components_.reshape((n_components, faces_h, faces_w)))

    combine_color_eigen_faces = combine_color_channels(color_eigen_faces)
    return combine_color_eigen_faces
# ------------------------------------------------------------------

def get_grey_scale_eigen_faces(faces, n_components):
    """
    Get grey scale eigenfaces.
    :return: array of grey scale eigenfaces
    """
    grey_faces = []
    for face in faces:
        grey_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        grey_faces.append(grey_face)
    grey_faces = np.array(grey_faces)

    grey_faces = grey_faces.reshape(grey_faces.shape[0], grey_faces.shape[1] * grey_faces.shape[2])
    pca = PCA(n_components=n_components, whiten=False).fit(grey_faces)
    eigen_faces = pca.components_.reshape((n_components, faces_h, faces_w))
    return eigen_faces
# ------------------------------------------------------------------

def pca_faces(faces_folder, color=True, n_components=9):
    """
    perform pca on cropped faces and plot eigenfaces.
    :param faces_folder: folder with uniform size cropped faces
    :param color: plot with color or grey scale
    :param n_components: The number of principal components for pca.
    :return: None
    """

    faces_files = glob.glob(faces_folder + "/*.jpg")
    faces = np.array([imread(fname) for fname in faces_files])
    if color:
        eigen_faces = get_color_eigen_faces(faces, n_components)
    else:
        eigen_faces = get_grey_scale_eigen_faces(faces, n_components)

    plot_eigenfaces(eigen_faces, faces_h, faces_w, color)

    # first save. show() resets the figure
    fname = faces_folder + "/eigen_faces"
    fname += "_color" if color else "_grey"
    plt.savefig(fname)

    plt.show()
# ------------------------------------------------------------------

def resize_face_images(faces_folder):
    faces_files = glob.glob(faces_folder + "/*.jpg")
    for face in faces_files:
        im = cv2.imread(face)
        im = cv2.resize(im, (faces_h, faces_w))
        cv2.imwrite(face, im)
# ------------------------------------------------------------------

def crop_faces(image_path, outout_folder, min_size=400):
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image_path)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe,
                                    scaleFactor=1.2,
                                    minNeighbors=5,
                                    minSize=(min_size, min_size)
                                    )

    img_filename = os.path.basename(image_path)
    for f in faces:
        x, y, w, h = [ v for v in f ]

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        face_file_name = outout_folder + "/" + img_filename.replace(".jpg", "_face.jpg")
        cv2.imwrite(face_file_name, sub_face)

#------------------------------------------------------------------

def main(imgs_folder, color):


    output_folder = imgs_folder + "/cropped_faces"
    os.makedirs(output_folder, exist_ok=True)

    # get images from input folder
    imgs_filenames = [os.path.basename(img) for img in glob.glob(imgs_folder + "/*.jpg")]
    if len(imgs_filenames) == 0:
        print("Couldn't find images in " + imgs_folder + " folder.")
        return

    # crop faces from images and save faces in output_folder
    for img in imgs_filenames:
        print(img)
        # min face size: 200X200 px
        # cropped faces are sved in output folder
        crop_faces(imgs_folder + "/" + img, output_folder, min_size=200)

    cropped_faces_folder = output_folder
    if len(glob.glob(cropped_faces_folder + "/*.jpg")) == 0:
        print("No faces we're found. try changing min_size or get other pictures")
        return

    # resize all cropped faces to the same size for pca
    resize_face_images(cropped_faces_folder)

    # ask the user to manually delete some non-faces crops
    input("Go clean " + os.path.basename(cropped_faces_folder) + " folder from non-face images.\nPress Enter after you're done.")

    # perform pca and plot eigenfaces
    pca_faces(cropped_faces_folder, color=color, n_components=9)

# ------------------------------------------------------------------

if __name__  == '__main__':

    if len(sys.argv) < 2:
        print("Usage: drag n drop a folder on the dragndrop.bat script "
              "\nOr give the argument via cmd line: <path to images folder> [optional 'grey' for grey scale output]\n")
        imgs_folder = "imgs"
    else:
        if os.path.exists(sys.argv[1]):
            imgs_folder = sys.argv[1]
        else:
            print("Enter a path to a folder with jpg images")
            sys.exit(0)

    if "grey" in sys.argv:
        color = False
    else:
        color = True

    main(imgs_folder, color)

