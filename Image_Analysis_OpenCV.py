if __name__=='__main__':
    import imageio
    import matplotlib
    import matplotlib.pyplot as plt
    import logging

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
    pic=imageio.imread('demo_2.jpg')
    plt.figure(figsize=(15,15))

    plt.imshow(pic)
    # plt.show() # optional

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
    Pegar um pixel específico localizado na 100a linha e 50a coluna
    Ver o RGB
'''

pic[100,50]
