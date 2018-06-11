import numpy as np
import math
import struct

class Data:
    def __init__(self, training_imgs_file="data/train-images-idx3-ubyte", training_lbls_file="data/train-labels-idx1-ubyte", test_imgs_file="data/t10k-images-idx3-ubyte", test_lbls_file="data/t10k-labels-idx1-ubyte", batch_size=100):
        self.default_batch_size = batch_size
        self.training_imgs_file = training_imgs_file
        self.training_lbls_file = training_lbls_file
        self.test_imgs_file = test_imgs_file
        self.test_lbls_file = test_lbls_file
        self.training_imgs, self.training_lbls = self.load_training_data()
        self.test_imgs, self.test_lbls = self.load_test_data()
        self.generate_label_logits()

    """take binary image file, and load the images into an ndarray"""
    def format_imgs(self, data_file):
        try:
            magic_num = struct.unpack(">L", data_file.read(4))[0] # magic number isn't used, but has some info about the file
            num_imgs = struct.unpack(">L", data_file.read(4))[0] # number of total images
            rows = struct.unpack(">L", data_file.read(4))[0] # per image
            cols = struct.unpack(">L", data_file.read(4))[0] # per image

            img_buffer = data_file.read(num_imgs * rows * cols) # reads all the data for all the images
            dt = np.dtype(np.uint8).newbyteorder('>') # big endian byte order
            img_array = np.frombuffer(img_buffer, dtype=dt, count=-1, offset=0) # make a one dimensional array of all the data
            img_array = np.reshape(img_array, (num_imgs, rows * cols)).transpose() # reshape array so that each column is an image
            img_array = img_array.astype(dtype=np.float32, casting='safe') # change data type to float32
        finally:
            return img_array

    """take binary label file, and load the labels into an ndarray"""
    def format_lbls(self, data_file):
        magic_num = struct.unpack(">L", data_file.read(4))[0] # not used(see above)
        num_lbls = struct.unpack(">L", data_file.read(4))[0] # total number of labels(same as number of images)
        try:
            lbl_buffer = data_file.read(num_lbls) # reads all the data for all the images
            dt = np.dtype(np.uint8).newbyteorder('>') # big endian byte order
            lbl_array = np.frombuffer(lbl_buffer, dtype=dt, count=-1, offset=0) # one d array with all images
            lbl_array = lbl_array.astype(dtype=np.float32, casting='safe') # change data type to float32
        finally:
            return lbl_array

    def generate_label_logits(self):
        self.training_lbls_logits = np.zeros((10, self.training_lbls.size), dtype=np.float32)
        for i in range(self.training_lbls.size):
            self.training_lbls_logits[int(self.training_lbls[i])][i] = 1
        self.test_lbls_logits = np.zeros((10, self.test_lbls.size))
        for i in range(self.test_lbls.size):
            self.test_lbls_logits[int(self.test_lbls[i])][i] = 1

    """perform zero mean, and unit variance normalization on images"""
    def normalize_imgs(self, imgs):
        mean = np.mean(imgs) # calculates the mean of all the pixels of all the images
        std = np.std(imgs) # calculates the standard deviation of all the pixels
        return (imgs - mean) / std # centers the values around zero, and devides by the deviation

    """randomize input images and labels(they will still line up)"""
    def randomize_data(self, imgs, lbls):
        permutation = np.random.permutation(imgs.shape[1]) # make a permutation of the indices of the images/labels
        shuffled_imgs = np.take(imgs, permutation, axis=-1) # apply the permutation to the images
        shuffled_lbls = np.take(lbls, permutation, axis=-1) # apply the permutation to the labels
        return shuffled_imgs, shuffled_lbls

    """display image for visualization purposes"""
    def display_image(self, img):
        disp = ['.', ',', ';', 'x'] # index of symbols
        image = img.reshape(785) # select one image to use
        for y in range(28):
            for x in range(28):
                symbol = disp[min(math.floor(image[y*28 + x]*(len(disp))), 3)] # determine the symbol to use for the current pixel
                print(symbol + symbol, end="") # display two of the symbols, to make the image square
            print("", end="\n") # start new line

    """load all the training images and labels"""
    def load_training_data(self, randomize=True):
        # open the raw data files
        training_imgs_raw = open(self.training_imgs_file, "rb")
        training_lbls_raw = open(self.training_lbls_file, "rb")

        # create numpy arrays with the raw data
        training_imgs = self.format_imgs(training_imgs_raw)
        training_lbls = self.format_lbls(training_lbls_raw)

        # close input streams
        training_imgs_raw.close()
        training_lbls_raw.close()

        # process the data, and prepare it for training
        training_imgs = self.normalize_imgs(training_imgs)
        training_imgs = np.append(training_imgs, np.ones((1, training_imgs.shape[1])), axis=0)

        #randomize
        if randomize:
            training_imgs, training_lbls = self.randomize_data(training_imgs, training_lbls)

        # return the processed data
        return training_imgs, training_lbls

    """load all the test images and labels"""
    def load_test_data(self, randomize=True):
        # open the raw data files
        test_imgs_raw = open(self.test_imgs_file, "rb")
        test_lbls_raw = open(self.test_lbls_file, "rb")

        # create numpy arrays with the correct (raw) data
        test_imgs = self.format_imgs(test_imgs_raw)
        test_lbls = self.format_lbls(test_lbls_raw)

        # close input streams
        test_imgs_raw.close()
        test_lbls_raw.close()

        # process the data, and prepare it for training
        test_imgs = self.normalize_imgs(test_imgs)
        test_imgs = np.append(test_imgs, np.ones((1, test_imgs.shape[1])), axis=0)

        #randomize
        if randomize:
            test_imgs, test_lbls = self.randomize_data(test_imgs, test_lbls)

        # return the processed data
        return test_imgs, test_lbls

    """create a random batch of images with specified size"""
    def next_batch(self, size=-2):
        if size == -1:
            return self.training_imgs, self.training_lbls, self.training_lbls_logits
        elif size == -2:
            size = self.default_batch_size
        indices = np.arange(self.training_imgs.shape[1])
        np.random.shuffle(indices)
        return np.take(self.training_imgs, indices[:size], axis=-1), np.take(self.training_lbls, indices[:size], axis=-1), np.take(self.training_lbls_logits, indices[:size], axis=-1) # use the permutation as indices to select random image-label pairs