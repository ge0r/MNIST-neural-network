import struct
import numpy as np
import os


def extract_data():

    train_img_path = os.path.abspath(r"data/train-images.idx3-ubyte")
    train_lbl_path = os.path.abspath(r"data/train-labels.idx1-ubyte")
    test_img_path = os.path.abspath(r"data/t10k-images.idx3-ubyte")
    test_lbl_path = os.path.abspath(r"data/t10k-labels.idx1-ubyte")

    with open(train_lbl_path, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        train_lbl = np.fromfile(flbl, dtype=np.int8)

    with open(train_img_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        train_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_lbl), rows, cols)

    with open(test_lbl_path, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        test_lbl = np.fromfile(flbl, dtype=np.int8)

    with open(test_img_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        test_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_lbl), rows, cols)

    return train_lbl, train_img, test_lbl, test_img


def show_image(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def prepare_labels(labels):
    output_labels = []

    for count, label in enumerate(labels):
        output_labels.append(np.zeros((10, 1)))
        output_labels[-1][label] = 1

    return output_labels


def prepare_data():
    train_lbl, train_img, test_lbl, test_img = extract_data()
    train_images = [np.reshape(x, (784, 1)) for x in train_img]
    test_images = [np.reshape(x, (784, 1)) for x in test_img]

    train_labels = prepare_labels(train_lbl)
    test_labels = prepare_labels(test_lbl)

    return train_images, test_images, train_labels, test_labels


