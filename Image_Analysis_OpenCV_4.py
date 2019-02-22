'''
    others functionalities with kernels
'''

# First we’ll measure distance from center of the image to every border pixel values
# And we take a convenient radius value and then using logical operator we’ll create a circular disc

if __name__=='__main__':
    import imageio
    import matplotlib
    import matplotlib.pyplot as plt
    import logging
    import numpy as np
    import random
    import warnings
    import matplotlib.cbook
    from skimage import color
    from skimage import exposure
    from scipy.signal import convolve2d
    from scipy.ndimage import (median_filter, gaussian_filter)
    from sklearn import cluster

    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    # configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.getLogger(__name__)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient.discovery').setLevel(logging.CRITICAL)
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
    logging.getLogger('oauth2client').setLevel(logging.WARNING)

    # read and plot
    pic=imageio.imread('demo_3.jpeg')
    plt.figure(figsize=(15,15))

    plt.imshow(pic)
    plt.axis('off')
    plt.show() # optional

    # Convert the image to greyscale
    img = color.rgb2grey(pic)

    # outlier kernel for edgedetection
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    # we use 'valid' wich means we do not add zero padding to our image
    edges = convolve2d(img, kernel, mode = 'valid')

    # Adjust the contrast of the filtered image by applying Histogram Equalization
    edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)),clip_limit = 0.03)

    # plot the edges_clipped
    plt.figure(figsize = (5,5))
    plt.imshow(edges_equalized, cmap='gray')
    img2 = plt.imshow(edges_equalized, cmap='gray')
    plt.axis('off')
    plt.show()

    '''
    using gaussian windows filter
    '''

    pic=imageio.imread('demo_3.jpeg')
    # Convert the image to grayscale
    img = color.rgb2gray(pic)
    # gaussian kernel - used for blurring
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
    kernel = kernel / np.sum(kernel)
    # we use 'valid' which means we do not add zero padding to our image
    edges = convolve2d(img, kernel, mode = 'valid')
    # Adjust the contrast of the filtered image by applying Histogram Equalization
    edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit = 0.03)
    # plot the edges_clipped
    plt.figure(figsize = (5,5))
    plt.imshow(edges_equalized, cmap='gray')
    plt.axis('off')
    plt.show()

    '''
    Sobel Kernels
    The Sobel kernels are used to show only the differences in adjacent pixel values in a particular direction.
    It tries to approximate the gradients of the image along one direction using kernel functions.
    '''

    pic=imageio.imread('demo_3.jpeg')

    # right sobel
    sobel_x = np.c_[[-1,0,1],[-2,0,2],[-1,0,1]]
    # top sobel
    sobel_y = np.c_[[1,2,1],[0,0,0],[-1,-2,-1]]
    ims = []
    for i in range(3):
        sx = convolve2d(pic[:,:,i], sobel_x, mode="same", boundary="symm")
        sy = convolve2d(pic[:,:,i], sobel_y, mode="same", boundary="symm")
        ims.append(np.sqrt(sx*sx + sy*sy))

    img_conv = np.stack(ims, axis=2).astype("uint8")

    plt.figure(figsize = (6,5))
    plt.axis('off')
    plt.imshow(img_conv)
    plt.show()

    '''
    To reduce noise. we generally use a filter like the Gaussian Filter, which is a digital filtering technique that is often used to remove noise from an image.
    '''

    def gaussain_filter_(img):
        """
        Applies a median filer to all channels
        """
        ims = []
        for d in range(3):
            img_conv_d = gaussian_filter(img[:,:,d], sigma = 4)
            ims.append(img_conv_d)
        return np.stack(ims, axis=2).astype("uint8")

    filtered_img = gaussain_filter_(pic)

    # right sobel
    sobel_x = np.c_[[-1,0,1],[-2,0,2],[-1,0,1]]
    # top sobel
    sobel_y = np.c_[[1,2,1],[0,0,0],[-1,-2,-1]]

    ims = []
    for d in range(3):
        sx = convolve2d(filtered_img[:,:,d], sobel_x, mode="same", boundary="symm")
        sy = convolve2d(filtered_img[:,:,d], sobel_y, mode="same", boundary="symm")
        ims.append(np.sqrt(sx*sx + sy*sy))

    img_conv = np.stack(ims, axis=2).astype("uint8")

    plt.figure(figsize=(7,7))
    plt.axis('off')
    plt.imshow(img_conv)
    plt.show()

    '''
    using Median Filter
    '''
    '''
    def median_filter_(img, mask):
        """
        Applies a median filer to all channels
        """
        ims = []
        for d in range(3):
            img_conv_d = median_filter(img[:,:,d], size=(mask,mask))
            ims.append(img_conv_d)
        return np.stack(ims, axis=2).astype("uint8")

    filtered_img = median_filter_(pic, 80)
    # right sobel
    sobel_x = np.c_[[-1,0,1],[-2,0,2],[-1,0,1]]
    # top sobel
    sobel_y = np.c_[[1,2,1],[0,0,0],[-1,-2,-1]]

    ims = []

    for d in range(3):
        sx = convolve2d(filtered_img[:,:,d], sobel_x, mode="same", boundary="symm")
        sy = convolve2d(filtered_img[:,:,d], sobel_y, mode="same", boundary="symm")
        ims.append(np.sqrt(sx*sx + sy*sy))

    img_conv = np.stack(ims, axis=2).astype("uint8")
    plt.figure(figsize=(7,7))
    plt.axis('off')
    plt.imshow(img_conv)
    plt.show()
    '''

    '''
    Clusterization K-means
    '''

    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from skimage import measure
    import numpy as np
    import imageio

    pic = imageio.imread('demo_3.jpeg')
    h,w = pic.shape[:2]
    im_small_long = pic.reshape((h * w, 3))
    im_small_wide = im_small_long.reshape((h,w,3))
    km = KMeans(n_clusters=2)
    km.fit(im_small_long)
    seg = np.asarray([(1 if i == 1 else 0)for i in km.labels_]).reshape((h,w))

    contours = measure.find_contours(seg, 0.5, fully_connected="high")
    simplified_contours = [measure.approximate_polygon(c, tolerance=5) for c in contours]

    plt.figure(figsize=(5,10))
    for n, contour in enumerate(simplified_contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.ylim(h,0)
    plt.axes().set_aspect('equal')
    plt.axis('off')
    plt.show()
