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
import multiprocessing as mp
import warnings
from functools import partial
import sys


class NumpyArrayEncoder(JSONEncoder):
    """
    Overwrite some of the json encoder to handle slices and numpy arrays.
    This allows dictionaries to be saved as json and loaded in later.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, slice):
            return [obj.start, obj.stop, obj.step]
        return JSONEncoder.default(self, obj)


def load_fits_data(filepath):
    """
    Load data/header from .fits file
    :param filepath:
    :return: header, data
    """

    hdul = fits.open(filepath)
    hdr = hdul[0].header
    data = hdul[2].data
    hdul.close()
    return hdr, data


def load_all_data(folder_path):
    """
    Load in all .fits files from a given directory
    :param folder_path:
    :return: data, hdr
    """

    hdr = []
    data = []
    for file in sorted(glob.glob(folder_path + '*.fits')):
        tmp = load_fits_data(file)
        hdr.append(tmp[0])
        data.append(tmp[1])

    data = np.asarray(data)
    return data, hdr


def filter_image_data(data):
    """
    Isolate speckles from image data. Most likely needs to already be of just an roi.
    TUNING PARAMETERS:    segmentation.flood(..., tolerance), very finicky
                            mophology.remove_small_objects(min_size)
    :param data: roi image
    :return: input image but non-speckle features are 0
    """
    erode = morphology.erosion(data)  # remove small background noise
    sobel = filters.sobel(erode)  # edge detection
    coords = np.unravel_index(np.argmin(erode), erode.shape)
    flood = np.invert(segmentation.flood(sobel, coords, tolerance=0.0005))  # fill to create mask for speckle only
    mask = morphology.remove_small_objects(flood, min_size=10)  # clean up small mask bits
    return data * mask


def label_image(img):
    """
    label each section of the image
    :param img:
    :return: original image but labelled
    """
    bool_img = morphology.closing(
        img.astype(bool))  # Connectivity is defined by having same value so need to convert to bool
    label_image = label(bool_img)
    return label_image


def cluster_single_speckle_kmeans(img, SPECKLE_SIZE):
    """
    cluster points using kmeans algorithm. includes both location and value of points
    :param img: roi img
    :param SPECKLE_SIZE: number of pixels in a speckle
    :return: kmeans clustering of points
    """
    points = np.asarray(np.where(img)).T
    weights = img[np.where(img)] / np.linalg.norm(img)
    N_clusters = round(np.sqrt(points.shape[0]) / SPECKLE_SIZE)

    kmeans = KMeans(n_clusters=N_clusters).fit(points, weights)
    return kmeans, points


def get_all_kmeans(img_label, regions, img, SPECKLE_SIZE):
    """
    get points using kmeans algorithm for all regions within the img
    :param img_label: labelled image
    :param img: roi image
    :param SPECKLE_SIZE: number of pixels in a speckle
    :return: kmeans_list: kmeans clustering of points for all regions of the image
             points: list of all cluster points (only used for plotting)
    """
    kmeans_list = []

    for label_index, region in enumerate(regions):
        if np.sqrt(region.area) < SPECKLE_SIZE:
            continue

        speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1, img_label),
                                       0) * img  # Isolate a single connected region

        single_kmeans, points = cluster_single_speckle_kmeans(speckle_cluster,
                                                              SPECKLE_SIZE)  # use kmeans to determine number of speckles in connected region

        kmeans_list.append(single_kmeans)
    return kmeans_list, points


def get_db_points(kmeans):
    """
    Cluster points using DBSCAN (density based) algorithm.
    TUNING PARAMETERS:  eps - finicky
                        min_sample
    :param kmeans: kmeans object for the image
    :return: single point for each speckle
    """
    test_points = []
    cluster_centers = [kmeans_iter.cluster_centers_ for kmeans_iter in kmeans]
    for cluster_center in cluster_centers:
        test_points.append(cluster_center)

    test_points = np.squeeze(test_points)
    try:
        db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
        # db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
    except:
        db_points = []
    return db_points


def cluster_data(img, img_label, regions, SPECKLE_SIZE):
    test_points = []
    for label_index, region in enumerate(regions):
        if np.sqrt(region.area) < SPECKLE_SIZE: continue

        speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1, img_label), 0) * img  # Isolate a single connected region


        kmeans, points = cluster_single_speckle_kmeans(speckle_cluster, SPECKLE_SIZE)  # use kmeans to determine number of speckles in connected region

        cluster_centers = kmeans.cluster_centers_

        for cluster_center in cluster_centers:
            test_points.append(cluster_center)

    test_points = np.asarray(test_points)
    db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
    db_points = test_points[db.core_sample_indices_]
    db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]

    '''
    try:
        db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
        db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
    except:
        db_points = []
    '''
    return kmeans, db_points


def worker(iter_img):
    """
    Multiprocess safe implementation of main processing. Takes raw img, applies roi, filters out non-speckles,
    labels processed image, clusters using kmeans and dbscan.
    :param iter_img:
    :param hdr:
    :return: dictionary with useful parameters (see bottom of function for list)
    """
    roi = (slice(180, 235), slice(150, 270))
    origin = (270, 329)
    SPECKLE_SIZE = 3.7  # 5.8 is calculated with 10um pinhole

    img = iter_img[roi]

    speckle_filter = filter_image_data(img)  # isolate speckles
    img_label = label_image(speckle_filter)  # label image
    regions = regionprops(img_label)  # get an array of regions with properties
    kmeans, db_points = cluster_data(img, img_label, regions, SPECKLE_SIZE)

    mean = np.mean(db_points, axis=0) + [roi[0].start, roi[1].start]
    std = np.std(db_points, axis=0)
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            mean = np.mean(db_points, axis=0) + [roi[0].start, roi[1].start]
            std = np.std(db_points, axis=0)
        except:
            mean = [0, 0]
            std = [0, 0]
    '''
    dx = origin[0] - mean[1]
    dy = origin[1] - mean[0]
    r = (np.sqrt(dx ** 2 + dy ** 2))
    phi = np.arctan(dy / dx)

    out = {}
    #out['regions'] = regions
    out['filtered_image'] = speckle_filter
    out['kmeans'] = kmeans
    out['label_image'] = img_label
    out['original_image'] = iter_img
    out['roi'] = roi
    out['roi_image'] = iter_img[roi]
    out['origin'] = origin
    out['db_points'] = db_points
    out['mean'] = mean
    out['std'] = std
    out['r'] = r
    out['phi'] = phi
    out['speckle_size'] = SPECKLE_SIZE
    return out


def make_figure(out_dict):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

    img = out_dict['roi_img']
    orig_img = out_dict['original_image']
    img_label = label_image(out_dict['filtered_image'])
    db_points = out_dict['db_points']
    origin = out_dict['origin']
    mean = out_dict['mean']
    speckle_filter = out_dict['filtered_image']
    image_label_overlay = label2rgb(out_dict['label_image'])
    r = out_dict['r']
    phi = out_dict['phi']
    roi = out_dict['roi']

    dx = roi[0].stop - roi[0].start
    dy = roi[1].stop - roi[1].start

    for label_index, region in enumerate(out_dict['regions']):
        points = [kpoints for kpoints in out_dict['kmeans_points']]
        klabels = [kmeans.labels_ for kmeans in out_dict['kmeans']]
        cluster_centers = [kmeans.cluster_centers_ for kmeans in out_dict['kmeans']]

        num_unique_labels = np.unique(klabels)
        n_clusters_ = len(set(klabels)) - (1 if -1 in klabels else 0)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(num_unique_labels))]

        ax[1, 0].scatter(regionprops(img_label)[label_index].centroid[1], regionprops(img_label)[label_index].centroid[0],
                         c='g', zorder=2)

        for k, col in zip(range(len(colors)), colors):
            my_members = klabels == k
            ax[1, 1].plot(points[my_members, 1], points[my_members, 0], '.', color=col, markersize=8, alpha=1, zorder=2)
            ax[1, 1].plot(cluster_centers[k, 1], cluster_centers[k, 0], '.', color=col, markersize=12, markeredgecolor='k',
                          zorder=2)

    ax[1, 2].plot(db_points[:, 1], db_points[:, 0], '.', color='r', markersize=12, markeredgecolor='k', zorder=2)

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
    return


if __name__ == '__main__':
    data, hdr = load_all_data('G:/My Drive/Data/FeGe_jumps/158K/2021 12 12/Andor DO436 CCD/')

    with mp.Pool(processes=mp.cpu_count()) as pool:
        out = pool.map(worker, data[:20], chunksize=1)

    for i, _ in enumerate(out):
        out[i]['hdr'] = hdr[i]
    '''
    make_figure(out[0])
    '''
    print('done')
