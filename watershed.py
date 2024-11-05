import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def preprocess(img: np.ndarray) -> np.ndarray:
    blur_img = cv.GaussianBlur(img, (5, 5), 0)
    thresh_img = cv.threshold(blur_img, 35, 255, cv.THRESH_BINARY)[1]
    open_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, np.ones((5, 5)))
    close_img = cv.morphologyEx(open_img, cv.MORPH_CLOSE, np.ones((5, 5)))

    return close_img


def find_markers(img: np.ndarray, footprint_size):
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, min_distance=20, footprint=np.ones((footprint_size, footprint_size)), labels=img)
    peaks = np.zeros_like(img)
    peaks[tuple(local_maxi.T)] = 255
    markers = ndi.label(peaks)[0]

    return distance, markers


def color_segments_find_contours_fill_dataframe(img_RGB, labels):
    colored_segments = np.zeros_like(img_RGB)
    img_RGB_with_contours = img_RGB.copy()
    df = pd.DataFrame(index=list(range(1, labels.max() + 1)), columns=['Moyenne de R', 'Moyenne de G', 'Moyenne de B'])
    for i in range(1, labels.max() + 1):
        # Color each segment
        colored_segments[labels == i] = np.random.randint(50, 220, 3)

        # Find contours for each segment
        single_segment_mask = (labels == i).astype(np.uint8)
        contour = cv.findContours(single_segment_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        # If the contour is too small area, skip it
        if cv.contourArea(contour[0]) < 100:
            continue

        # Draw the contours on the image
        cv.drawContours(img_RGB_with_contours, contour  , -1, (0, 255, 0), 2)

        # add the label to the image for each segment
        # find the center of the segment
        x = 0
        y = 0
        for point in contour[0]:
            x += point[0][0]
            y += point[0][1]
        x = int(x / len(contour[0])) - 15
        y = int(y / len(contour[0])) + 15
        cv.putText(img_RGB_with_contours, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
        cv.putText(colored_segments, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)

        # fill the dataframe with the mean of each channel for each segment
        mean_R = np.mean(img_RGB[labels == i][0])
        mean_G = np.mean(img_RGB[labels == i][1])
        mean_B = np.mean(img_RGB[labels == i][2])
        df.loc[i] = [mean_R, mean_G, mean_B]

    return img_RGB_with_contours, colored_segments, df



def my_watershed(img_path: str, local_peak_kernel_size:int=53) -> (np.ndarray, np.ndarray, pd.DataFrame, int):
    """
    :param img_path:
    :param local_peak_kernel_size: adjust the number of local peaks / segments
    :return: the original image with the contours of the segments, an image with the segments colored and a black background, the number of segments
    """
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_RGB = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    preprocessed_img = preprocess(img)

    distance, markers = find_markers(preprocessed_img, local_peak_kernel_size)

    labels = watershed(-distance, markers, mask=preprocessed_img)

    contours, segments, df_mean = color_segments_find_contours_fill_dataframe(img_RGB, labels)

    return contours, segments, df_mean, labels.max()


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
            contours, segments, _, _ = my_watershed(f'Images/Echantillion1Mod2_{n}.png', local_peak_kernel_size=s)
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