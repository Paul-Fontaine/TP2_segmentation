import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def my_watershed(img_path, local_peak_kernel_size=51, ):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_RGB = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    blur_img = cv.GaussianBlur(img, (5, 5), 0)
    thresh_img = cv.threshold(blur_img, 35, 255, cv.THRESH_BINARY)[1]
    open_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, np.ones((5, 5)))
    close_img = cv.morphologyEx(open_img, cv.MORPH_CLOSE, np.ones((5, 5)))

    distance = ndi.distance_transform_edt(close_img)
    local_maxi = peak_local_max(distance,min_distance=20, footprint=np.ones((local_peak_kernel_size, local_peak_kernel_size)), labels=open_img)
    peaks = np.zeros_like(open_img)
    peaks[tuple(local_maxi.T)] = 255
    markers = ndi.label(peaks)[0]

    labels = watershed(-distance, markers, mask=close_img)

    colored_segments = np.zeros_like(img_RGB)
    img_RGB_with_contours = img_RGB.copy()
    for i in range(1, labels.max() + 1):
        # Color each segment
        colored_segments[labels == i] = np.random.randint(0, 255, 3)

        # Find contours for each segment
        single_segment_mask = (labels == i).astype(np.uint8)
        contours = cv.findContours(single_segment_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        # Draw the contours on the image
        cv.drawContours(img_RGB_with_contours, contours, -1, (0, 255, 0), 2)

    return img_RGB_with_contours, colored_segments, labels.max()+1


def search_footprint():
    nb_kayoux = np.zeros((9, 9), np.uint8)
    for i, s in enumerate(range(47, 64, 2)):
        for j, n in enumerate([301, 302, 303, 304, 305, 306, 316, 422, 471]):
            nb_kayoux[i, j] = my_watershed(f'Images/Echantillion1Mod2_{n}.png', local_peak_kernel_size=s)[2]
    np.savetxt('nb_kayoux.csv', nb_kayoux, delimiter=',', fmt='%i')
    means_nb_kayoux = np.mean(nb_kayoux, axis=1)
    for i, s in enumerate(range(47, 64, 2)):
        ecarts = nb_kayoux[i] - means_nb_kayoux[i]
        ecart_moyen = np.mean(ecarts)
        plt.figure(num=i, figsize=(20, 90))
        plt.suptitle(f'Local peak kernel size: {s}')
        for j, n in enumerate([301, 302, 303, 304, 305, 306, 316, 422, 471]):
            contours, segments, _ = my_watershed(f'Images/Echantillion1Mod2_{n}.png', local_peak_kernel_size=s)
            plt.subplot(9, 2, j * 2 + 1)
            plt.imshow(contours)
            plt.title(f'Nb kayoux: {nb_kayoux[i, j]}, Mean: {round(means_nb_kayoux[j], 2)}')
            plt.axis('off')
            plt.subplot(9, 2, j * 2 + 2)
            plt.imshow(segments)
            plt.axis('off')

        plt.show()


if __name__ == "__main__":
    search_footprint()
