import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import glob
from astropy.io import fits
import matplotlib as mpl
from skimage.measure import block_reduce
from skimage import filters, segmentation, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn import cluster
from sklearn.cluster import KMeans
import json
from json import JSONEncoder


def load_fits_data(filepath):
    # Load data/header from .fits file
    hdul = fits.open(filepath)
    hdr = hdul[0].header
    data = hdul[2].data
    hdul.close()
    return hdr, data


def load_all_data(folder_path):
    hdr = []
    data = []
    for file in sorted(glob.glob(folder_path + '*.fits')):
        tmp = load_fits_data(file)
        hdr.append(tmp[0])
        data.append(tmp[1])

    data = np.asarray(data)
    return data, hdr


def filter_image_data(data):
    erode = morphology.erosion(data) # remove small background noise
    sobel = filters.sobel(erode) # edge detection
    coords = np.unravel_index(np.argmin(erode), erode.shape)
    flood = np.invert(segmentation.flood(sobel, coords, tolerance=0.0005)) # fill to create mask for speckle only
    mask = morphology.remove_small_objects(flood, min_size=10) # clean up small mask bits
    return data*mask


def label_image(img):
    bool_img = morphology.closing(img.astype(bool)) # Connectivity is defined by having same value so need to convert to bool
    label_image = label(bool_img)
    #image_label_overlay = label2rgb(label_image, image=img, bg_label=0)
    #regions = regionprops(label_image)
    return label_image


def cluster_single_speckle_kmeans(img, SPECKLE_SIZE):
    points = np.asarray(np.where(img)).T
    weights = img[np.where(img)] / np.linalg.norm(img)
    N_clusters = round(np.sqrt(points.shape[0]) / SPECKLE_SIZE)

    kmeans = KMeans(n_clusters=N_clusters).fit(points, weights)
    return kmeans, points


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, slice):
            return [obj.start, obj.stop, obj.step]
        return JSONEncoder.default(self, obj)


if __name__ == '__main__':
    data, hdr = load_all_data('G:/My Drive/Data/FeGe_jumps/158K/2021 12 12/Andor DO436 CCD/')

    roi = (slice(180, 235), slice(150, 270))
    origin = (270, 329)
    SPECKLE_SIZE = 3.7  # 5.8 is calculated with 10um pinhole

    dx = roi[0].stop - roi[0].start
    dy = roi[1].stop - roi[1].start

    out = {}

    for i, iter_img in enumerate(data):
        if i > 2: break
        out[i] = {}
        orig_img = iter_img.copy()
        img = iter_img[roi].copy()

        speckle_filter = filter_image_data(img)  # isolate speckles
        img_label = label_image(speckle_filter)  # label image
        image_label_overlay = label2rgb(img_label, bg_label=0)  # convert regions to rgb (for plotting only)
        regions = regionprops(img_label)  # get an array of regions with properties

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

        test_points = []
        for label_index, region in enumerate(regions):
            if np.sqrt(region.area) < SPECKLE_SIZE: continue

            speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1, img_label),
                                           0) * img  # Isolate a single connected region

            kmeans, points = cluster_single_speckle_kmeans(speckle_cluster,
                                                           SPECKLE_SIZE)  # use kmeans to determine number of speckles in connected region

            klabels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            num_unique_labels = np.unique(klabels)
            n_clusters_ = len(set(klabels)) - (1 if -1 in klabels else 0)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(num_unique_labels))]

            ax[1, 0].scatter(regionprops(img_label)[label_index].centroid[1],
                             regionprops(img_label)[label_index].centroid[0], c='g', zorder=2)

            for k, col in zip(range(len(colors)), colors):
                my_members = klabels == k
                ax[1, 1].plot(points[my_members, 1], points[my_members, 0], '.', color=col, markersize=8, alpha=1, zorder=2)
                ax[1, 1].plot(cluster_centers[k, 1], cluster_centers[k, 0], '.', color=col, markersize=12,
                              markeredgecolor='k', zorder=2)

            for cluster_center in cluster_centers:
                test_points.append(cluster_center)

        test_points = np.asarray(test_points)

        try:
            db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
            db_points = test_points[db.core_sample_indices_]
            db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
            ax[1, 2].plot(db_points[:, 1], db_points[:, 0], '.', color='r', markersize=12, markeredgecolor='k', zorder=2)

        except:
            db_points = []

        mean = np.mean(db_points, axis=0) + [roi[0].start, roi[1].start]
        std = np.std(db_points, axis=0)
        dxx = origin[0] - mean[1]
        dyy = origin[1] - mean[0]
        r = (np.sqrt((dxx) ** 2 + (dyy) ** 2))
        phi = np.arctan(dyy / dxx)

        out[i]['points'] = db_points
        out[i]['mean'] = mean
        out[i]['std'] = std
        out[i]['r'] = r
        out[i]['phi'] = phi
        out[i]['Iz'] = hdr[i]['Iz']
        out[i]['speckle_size'] = SPECKLE_SIZE
        out[i]['roi'] = roi
        out[i]['origin'] = origin

        ax[0, 0].imshow(orig_img, norm=mpl.colors.LogNorm())

        ax[0, 0].plot(origin[0], origin[1], 'x', color='r', markersize=8)
        ax[0, 0].plot(mean[1], mean[0], 'x', color='r', markersize=8)
        ax[0, 0].add_patch(
            mpl.patches.Arrow(origin[0], origin[1], dx=-r * np.cos(phi), dy=-r * np.sin(phi), edgecolor='b', facecolor='b',
                              zorder=2))

        ax[0, 0].add_patch(
            mpl.patches.Rectangle((roi[1].start, roi[0].start), dy, dx, linewidth=1, edgecolor='r', facecolor='none'))
        ax[0, 1].imshow(img, zorder=1)
        ax[0, 2].imshow(speckle_filter)
        ax[1, 0].imshow(image_label_overlay)

        ax[1, 1].imshow(img, zorder=1)

        ax[1, 2].imshow(img, zorder=1)

        plt.tight_layout()
        plt.show()

