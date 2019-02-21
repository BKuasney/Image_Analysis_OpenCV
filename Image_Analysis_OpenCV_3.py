'''
    MASK
    Image mask is an image processing technique that is used to removethe background
    Now, we will create a mask that is in shape of a circular disk
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

    '''
    negative image
    '''

    negative = 255 - pic
    plt.figure(figsize= (6,6))
    plt.imshow(negative)
    plt.axis('off')
    plt.show()

    '''
    log transformation
    During log transformation, the dark pixels in an image are expanded as compared to the higher pixel values. The higher pixel values are kind of compressed in log transformation
    '''
    gray= lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])
    gray=gray(pic)

    max_=np.max(gray)

    def log_transform():
        return(255/np.log(1+max_))*np.log(1+gray)

    plt.figure(figsize=(5,5))
    plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
    plt.axis('off');
    plt.show()

    '''
    gama transformation
    Gamma correction, or often simply gamma, is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems
    '''
    gamma=2.2# Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright

    gamma_correction=((pic/255)**(1/gamma))
    plt.figure(figsize=(5,5))
    plt.imshow(gamma_correction)
    plt.axis('off')
    plt.show()

    '''
    Convolution (using kernel)
    An image kernel or filter is a small matrix used to apply effects like the ones we might find in Photoshop or Gimp, such as blurring, sharpening, outlining or embossing
    They’re also used in machine learning for feature extraction, a technique for determining the most important portions of an image
    https://en.wikipedia.org/wiki/Kernel_(image_processing)#Details
    '''

    from scipy.signal import convolve2d

    def Convolution(image, kernel):
        conv_bucket = []
        for d in range(image.ndim):
            conv_channel = convolve2d(image[:,:,d], kernel,
                                mode="same", boundary="symm")
            conv_bucket.append(conv_channel)
        return np.stack(conv_bucket, axis=2).astype("uint8")

    kernel_sizes = [9,15,30,60]
    fig, axs = plt.subplots(nrows = 1, ncols = len(kernel_sizes), figsize=(15,15));

    pic = imageio.imread('demo_3.jpeg')

    for k, ax in zip(kernel_sizes, axs):
        kernel = np.ones((k,k))
        kernel /= np.sum(kernel)
        ax.imshow(Convolution(pic, kernel))
        ax.set_title("Convolved By Kernel: {}".format(k))
    ax.set_axis_off()
    plt.show()
