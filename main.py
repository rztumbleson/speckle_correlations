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


def get_all_kmeans(img_label, img, SPECKLE_SIZE):
    """
    get points using kmeans algorithm for all regions within the img
    :param img_label: labelled image
    :param img: roi image
    :param SPECKLE_SIZE: number of pixels in a speckle
    :return: kmeans clustering of points for all regions of the image
    """
    kmeans_list = []
    regions = regionprops(img_label)  # get an array of regions with properties
    for label_index, region in enumerate(regions):
        if np.sqrt(region.area) < SPECKLE_SIZE:
            continue

        speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1, img_label),
                                       0) * img  # Isolate a single connected region

        single_kmeans, points = cluster_single_speckle_kmeans(speckle_cluster,
                                                              SPECKLE_SIZE)  # use kmeans to determine number of speckles in connected region

        kmeans_list.append(single_kmeans)
    return kmeans_list


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
        test_points.append(cluster_center.ravel())

    test_points = np.asarray(test_points)
    try:
        db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
        # db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
    except:
        db_points = []
    return db_points


def worker(iter_img, hdr):
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
    kmeans = get_all_kmeans(img_label, img, SPECKLE_SIZE)
    db_points = get_db_points(kmeans)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            mean = np.mean(db_points, axis=0) + [roi[0].start, roi[1].start]
            std = np.std(db_points, axis=0)
        except:
            mean = [0, 0]
            std = [0, 0]

    dx = origin[0] - mean[1]
    dy = origin[1] - mean[0]
    r = (np.sqrt(dx ** 2 + dy ** 2))
    phi = np.arctan(dy / dx)

    out = {}
    out['original_image'] = iter_img
    out['roi'] = roi
    out['roi_image'] = iter_img[roi]
    out['origin'] = origin
    out['db_points'] = db_points
    out['mean'] = mean
    out['std'] = std
    out['r'] = r
    out['phi'] = phi
    out['Iz'] = hdr['Iz']
    out['speckle_size'] = SPECKLE_SIZE
    return out


if __name__ == '__main__':
    data, hdr = load_all_data('G:/My Drive/Data/FeGe_jumps/158K/2021 12 12/Andor DO436 CCD/')
    print(mp.cpu_count())
    with mp.Pool(processes=mp.cpu_count()) as pool:
        out = pool.map(partial(worker, hdr=hdr), (data[:10]), chunksize=1)

    print('done')
