if __name__=='__main__':
    import imageio
    import matplotlib
    import matplotlib.pyplot as plt
    import logging
    import numpy as np

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

# Basic properties
logging.info('Type of the image: {}'.format(type(pic)))
logging.info('Shape of the image : {}'.format(pic.shape))
logging.info('Image Hight {}'.format(pic.shape[0]))
logging.info('Image Width {}'.format(pic.shape[1]))
logging.info('Dimension of Image {}'.format(pic.ndim))


logging.info('Image size {}'.format(pic.size))
logging.info('Maximum RGB value in this image {}'.format(pic.max()))
logging.info('Minimum RGB value in this image {}'.format(pic.min()))


'''
    Pick a specific pixel locale at 100th row and 50th column
    return the RGB
'''

# return the RGB from pick pixel
logging.info(pic[100,50])

# return only R, oly G and onle B
# remenber we dealing with 3 dimensions
logging.info('Value of R: {}'.format(pic[100,50,0]))
logging.info('Value of G: {}'.format(pic[100,50,1]))
logging.info('Value of B: {}'.format(pic[100,50,2]))

# now, we see each channel in whole image

# see the image only with R from the RGB
plt.title('R channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))
plt.imshow(pic[:,:,0])
plt.show()

# see the image only with G from the RGB
plt.title('G channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))
plt.imshow(pic[ : , : , 1])
plt.show()

# see the image only with B from the RGB
plt.title('B channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))
plt.imshow(pic[ : , : , 2])
plt.show()


#Y' = 0.299 R + 0.587 G + 0.114 B
pic=imageio.imread('demo_3.jpeg')
gray= lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)
plt.figure(figsize=(10,10))
plt.imshow(gray,cmap=plt.get_cmap(name='gray'))
plt.show()
