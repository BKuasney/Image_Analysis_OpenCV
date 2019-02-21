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
    plt.show() # optional

    # separete the row and column values
    total_row, total_col, layers = pic.shape

    #create a vector
    # Ogrid is a compact method of creating a multidimensional nd array operations in single lines
    x,y=np.ogrid[:total_row,:total_col]

    # get the center values of image
    cen_x, cen_y = total_row/2, total_col/2

    # Select convenient radius value
    radius = (total_row/2)

    # total distance from the center
    distance_from_the_center = np.sqrt((x-cen_x)**2+(y-cen_y)**2)

    # now, we will filter values less then the radius value
    circular_pic = distance_from_the_center>radius

    # let's assign value zero for all pixel value that outside the circular disk
    pic[circular_pic] = 0
    plt.figure(figsize=(10,10))
    plt.imshow(pic)
    plt.show()


    '''
    We can filter colors from the picture
    This approach is used in satellite Image Processing
    Where cientist can separeted aspects from the image to analyze
    '''

    # Only red pixel value, higher than 180
    pic=imageio.imread('demo_3.jpeg')
    red_mask=pic[:,:,0]<100
    pic[red_mask]=1
    plt.figure(figsize=(15,15))
    plt.imshow(pic)
    plt.show()
