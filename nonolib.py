import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # check alpha channel
    if image.shape[2] == 4:
        # make mask of where the transparent bits are
        trans_mask = image[:,:,3] == 0
        # replace areas of transparency with white and not transparent
        image[trans_mask] = [255, 255, 255, 255]
        # new image without alpha channel...
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def plot_image(image, gray=False, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if not gray:
        ax.imshow(image[...,::-1])
    else:
        ax.imshow(image, 'gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if ax is None:
        plt.show()


def squarify_image(image):
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, color_depth= image.shape

    if width > height:
        start = round((width - height) / 2)
        end = round(start + height)
        cropped_image = image[:, start:end]
    else:
        start = round((height - width) / 2)
        end = round(start + width)
        cropped_image = image[start:end, :]
    return cropped_image

def generate_nonogram(img_path, n_cells=15, method='canny-edge', return_colored_result=False):
    # load image
    image = load_image(img_path)
    # squarify
    image = squarify_image(image)
    
    if return_colored_result:
        colored_result = cv2.resize(image, (n_cells, n_cells))
    
    # method: grayscale
    if method == 'grayscale':
        # resize
        image = cv2.resize(image, (n_cells, n_cells))
        colored_result = image

        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # binarize
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif method == 'canny-edge':
        # edge detection using canny algorithm
        edges = cv2.Canny(image,100,200)
        
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.resize(edges, (n_cells, n_cells))

        ret, image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)

    if return_colored_result:
        return image, colored_result
    else:
        return image
    
def get_count_str(arr, col=False):
    res = []
    cnt = -1
    for elem in arr:
        if elem == 0:
            if cnt == -1:
                cnt = 1
            else:
                cnt += 1
        else:
            if cnt > 0:
                res.append(str(cnt))
                cnt = -1

    if cnt > 0:
        res.append(str(cnt))
    if res:
        if col:
            return '\n'.join(res)
        else:
            return ' '.join(res)
    else:
        return '0'


def create_nonogram_plot(nono_mat):
    
    n_cells = nono_mat.shape[0]
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    ax.set_xlim(0, n_cells)
    ax.set_ylim(0, n_cells)

    ax.axvline(0, color='black', lw=2)
    ax.axhline(0, color='black', lw=2)
    for i in range(0, n_cells):
        if n_cells%5 == 0 and i%5 == 0:
            line_width = 2
        else:
            line_width = 1
        ax.axvline(i, color='black', lw=line_width)
        ax.axhline(i, color='black', lw=line_width)

        # column and row labels
        row_label = get_count_str(nono_mat[i, :])
        col_label = get_count_str(nono_mat[:, i], col=True)
        ax.text(-0.2, i+0.5, row_label, horizontalalignment='right')
        ax.text(i+0.5, -0.2, col_label, horizontalalignment='center')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.invert_yaxis()