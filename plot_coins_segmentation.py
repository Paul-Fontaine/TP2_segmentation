import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage import data

coins = data.coins()

"""
Region-based segmentation
=========================

We therefore try a region-based method using the watershed transform. First, we
find an elevation map using the Sobel gradient of the image.
"""


elevation_map = sobel(coins)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(elevation_map, cmap='gray', interpolation='nearest')
ax.axis('off')
ax.set_title('elevation_map')

"""
Next we find markers of the background and the coins based on the extreme parts
of the histogram of grey values.
"""

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 160] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap='bone', interpolation='nearest')
# add a color bar with only the colors of interest
cmap = plt.cm.bone
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = [0, 1, 2]
norm = plt.Normalize(0, 2)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=bounds)
ax.axis('off')
ax.set_title('markers')

"""
.. image:: PLOT2RST.current_figure

Finally, we use the watershed transform to fill regions of the elevation map
starting from the markers determined above:

"""
segmentation = watershed(elevation_map, markers)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('segmentation')

"""
.. image:: PLOT2RST.current_figure

This last method works even better, and the coins can be segmented and labeled
individually.

"""


segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_coins, image=coins)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
ax1.imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
ax1.contour(segmentation, [0.5], linewidths=1.2, colors='y')
ax1.axis('off')
ax2.imshow(image_label_overlay, interpolation='nearest')
ax2.axis('off')

plt.show()
