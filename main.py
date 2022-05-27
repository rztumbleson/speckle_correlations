import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import glob
from astropy.io import fits
import matplotlib as mpl
import matplotlib.patches as mpatches
from skimage import filters, segmentation, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn import cluster
from sklearn.cluster import KMeans
import multiprocessing as mp
import warnings
import os
import pandas as pd
from tqdm import tqdm


def load_fits_data(filepath):
    """
    Load data/header from .fits file
    :param filepath:
    :return: header, data
    """

    try:
        with fits.open(filepath) as hdul:
            # Beamline 12
            hdr = hdul[0].header
            data = hdul[2].data
    except IndexError:
        with fits.open(filepath) as hdul:
            # Cosmic
            try:
                hdr = hdul[0].header
                data = hdul[0].data
            except IndexError:
                print(hdul.info())

    return data, hdr


def load_all_data(folder_path, n_files=None):
    """
    Load in all .fits files from a given directory
    :param folder_path:
    :return: data, hdr
    """

    hdr = []
    data = []
    for ii, file in tqdm(enumerate(sorted(glob.glob(folder_path + '*.fits'))), desc='loading data'):
        if n_files is not None:
            if n_files <= ii:
                break
        tmp = load_fits_data(file)
        hdr.append(tmp[1])
        data.append(tmp[0])

    data = np.asarray(data)

    print(f'Loaded data shape: {data.shape}')

    return np.squeeze(data), hdr


def filter_image_data(data, tol=0.000_000_005, min_size=10):
    """
    Isolate speckles from image data. Most likely needs to already be of just an roi.
    TUNING PARAMETERS:    segmentation.flood(..., tolerance), very finicky
                            mophology.remove_small_objects(min_size)
    :param data: roi image
    :return: input image but non-speckle features are 0
    """
    erode = morphology.erosion(data)  # remove small background noise
    sobel = filters.sobel(erode)  # edge detection
    coords = np.unravel_index(np.argmin(data), data.shape)
    flood = np.invert(segmentation.flood(sobel, coords, tolerance=tol))  # fill to create mask for speckle only
    mask = morphology.remove_small_objects(flood, min_size=min_size)  # clean up small mask bits
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


def cluster_single_speckle_kmeans(img, speckle_size):
    """
    cluster points using kmeans algorithm. includes both location and value of points
    :param img: roi img
    :param speckle_size: number of pixels in a speckle
    :return: kmeans clustering of points
    """
    points = np.asarray(np.where(img)).T
    weights = img[np.where(img)] / np.linalg.norm(img)
    N_clusters = round(np.sqrt(points.shape[0]) / speckle_size)

    kmeans = KMeans(n_clusters=N_clusters).fit(points, weights)
    return kmeans, points


def get_all_kmeans(img_label, regions, img, speckle_size):
    """
    CURRENTLY NOT USING
    get points using kmeans algorithm for all regions within the img
    :param img_label: labelled image
    :param img: roi image
    :param speckle_size: number of pixels in a speckle
    :return: kmeans_list: kmeans clustering of points for all regions of the image
             points: list of all cluster points (only used for plotting)
    """
    kmeans_list = []

    for label_index, region in enumerate(regions):
        if np.sqrt(region.area) < speckle_size:
            continue

        speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1, img_label),
                                       0) * img  # Isolate a single connected region

        single_kmeans, points = cluster_single_speckle_kmeans(speckle_cluster,
                                                              speckle_size)  # use kmeans to determine number of speckles in connected region

        kmeans_list.append(single_kmeans)
    return kmeans_list, points


def get_db_points(kmeans, eps=15, min_samples=2):
    """
    CURRENTLY NOT USING
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
        db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(test_points)
        # db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
    except:
        db_points = []
    return db_points


def cluster_data(img, img_label, regions, speckle_size, eps=15, min_samples=2):
    """
    cluster data using kmeans (both location and intensity) and then cluster kmeans cluster with dbscan (density-based).
    :param img: roi image
    :param img_label: labelled roi image
    :param regions: regions object from skimage.measure
    :param speckle_size: number of pixels in single speckle
    :return: kmeans_object: list of kmeans objects
            kmeans_points: list of points for each kmeans object
            dbpoints: list (possibly ndarray) of final point clustering
    """
    test_points = []
    kmeans_points = []
    kmeans_object = []
    for label_index, region in enumerate(regions):
        if np.sqrt(region.area) < speckle_size:
            continue

        speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1,
                                                          img_label), 0) * img  # Isolate a single connected region

        kmeans, points = cluster_single_speckle_kmeans(speckle_cluster,
                                                       speckle_size)  # determine number of speckles in connected region
        kmeans_points.append([tuple(p) for p in points])
        kmeans_object.append(kmeans)
        cluster_centers = kmeans.cluster_centers_

        for cc in cluster_centers:
            test_points.append(cc)

    test_points = np.asarray(test_points)
    try:
        db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(test_points)
        # db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
        #db_points = [tuple(dbp) for dbp in db_points]
    except ValueError:
        db_points = []

    return kmeans_object, kmeans_points, db_points


def worker(args):
    """
    Multiprocess safe implementation of main processing. Takes raw img, applies roi, filters out non-speckles,
    labels processed image, clusters using kmeans and dbscan.
    :param args: two tuple: args[0] single raw data image, args[1] hdr
    :return: dictionary with useful parameters (see bottom of function for list)
    """

    roi = (slice(190, 245), slice(150, 270))
    origin = (270, 329)
    speckle_size = 5.8  # 5.8 is calculated with 10um pinhole

    tol = 0.000_5
    min_size = 10
    eps = 15
    min_samples = 2

    '''
    roi = (slice(250, 750), slice(0, 300))
    origin = (578, 535)
    speckle_size = 7
    mask = np.load('G:/My Drive/Python/speckle_clustering/mask.npy')
    '''

    iter_img = args[0]
    hdr = args[1]
    img = iter_img[roi]
    #img[np.where(~mask.astype(bool))] = np.mean(img)


    speckle_filter = filter_image_data(img, tol=tol, min_size=min_size)  # isolate speckles
    img_label = label_image(speckle_filter)  # label image
    regions = regionprops(img_label)  # get an array of regions with properties
    kmeans, kmeans_points, db_points = cluster_data(img, img_label, regions, speckle_size,eps=eps, min_samples=min_samples)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            mean = np.mean(db_points, axis=0) + [roi[0].start, roi[1].start]

            dx = origin[0] - mean[1]
            dy = origin[1] - mean[0]
            r = (np.sqrt(dx ** 2 + dy ** 2))
            phi = np.arctan(dy / dx)

        except RuntimeWarning:
            # mean of an empty slice
            mean = [0, 0]
            r = None
            phi = None


    out = {}
    out['tolerence'] = tol
    out['min_size'] = min_size
    out['eps'] = eps
    out['min_samples'] = min_samples
    out['N_regions'] = len(kmeans)
    out['filtered_image'] = speckle_filter
    out['kmeans'] = kmeans
    out['kmeans_points'] = kmeans_points
    out['label_image'] = img_label
    out['original_image'] = iter_img
    out['roi'] = roi
    out['roi_image'] = iter_img[roi]
    out['origin'] = origin
    out['db_points'] = db_points
    out['db_mean'] = tuple(mean)
    out['r'] = r
    out['phi'] = phi
    out['speckle_size'] = speckle_size
    out['Iz'] = hdr['Iz']
    out['hdr'] = hdr

    return pd.DataFrame([out])


def make_figures(df, n_figures=None, show_image=True, save_image=False, save_path='./imgs/'):
    """
    2x3 figure summarizing image processing.
    upper left: original image showing roi, centroid point, origin, and line that defines r and phi
    upper middle: original image within roi
    upper right: filtered image within roi
    lower left: output of label2rgb for first label on filtered image
    lower middle: kmeans clustered points and centroids
    lower right: dbscan centroid of clusters from kmeans points (note: dbscan filters out some of the kmeans clusters)
    :param df: pandas data frame from worker function
    :param n_figures: (int) number of figures to plot/save
    :param show_image: (bool) plot each figure
    :param save_image: (bool) save image at save_path
    :param save_path: (str) string path of image output folder
    :return: matplotlib figure and axes
    """
    if n_figures is None:
        n_figures = df.shape[0]
    for ii in tqdm(range(n_figures), desc='making figures'):
        df_iter = df.loc[ii]
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))

        img = df_iter['roi_image']
        orig_img = df_iter['original_image']
        db_points = df_iter['db_points']
        origin = df_iter['origin']
        mean = df_iter['db_mean']
        speckle_filter = df_iter['filtered_image']
        image_label_overlay = label2rgb(df_iter['label_image'], bg_label=0)
        r = df_iter['r']
        phi = df_iter['phi']
        roi = df_iter['roi']

        dx = roi[0].stop - roi[0].start
        dy = roi[1].stop - roi[1].start

        for label_index in range(df_iter['N_regions']):
            points = np.asarray(df_iter['kmeans_points'][label_index])
            klabels = np.asarray(df_iter['kmeans'][label_index].labels_)
            cluster_centers = np.asarray(df_iter['kmeans'][label_index].cluster_centers_)

            num_unique_labels = np.unique(klabels)

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(num_unique_labels))]
            '''
            ax[1, 0].scatter(regionprops(img_label)[label_index].centroid[1],
                             regionprops(img_label)[label_index].centroid[0],
                             c='g', zorder=2)
            '''
            for k, col in zip(range(len(colors)), colors):
                my_members = klabels == k
                ax[1, 1].plot(points[my_members, 1], points[my_members, 0], '.', color=col, markersize=1,
                              alpha=1, zorder=2)
                ax[1, 1].plot(cluster_centers[k, 1], cluster_centers[k, 0], '.', color=col, markersize=12,
                              markeredgecolor='k', zorder=2)

        try:
            ax[1, 2].plot(db_points[:, 1], db_points[:, 0], '.', color='r', markersize=12,
                          markeredgecolor='k', zorder=2)

        except TypeError:
            # This happens if db_points = []
            pass

        ax[0, 0].imshow(orig_img, norm=mpl.colors.LogNorm())

        ax[0, 0].plot(origin[0], origin[1], 'x', color='r', markersize=8)
        ax[0, 0].plot(mean[1], mean[0], 'x', color='r', markersize=8)

        try:
            ax[0, 0].add_patch(
                mpatches.Arrow(origin[0], origin[1], dx=-r * np.cos(phi), dy=-r * np.sin(phi), edgecolor='b',
                               facecolor='b', zorder=2))
        except TypeError:
            # This happens if r and phi are None
            pass

        ax[0, 0].add_patch(
            mpatches.Rectangle((roi[1].start, roi[0].start), dy, dx, linewidth=1, edgecolor='r', facecolor='none'))
        ax[0, 1].imshow(img, zorder=1)
        ax[0, 2].imshow(speckle_filter)
        ax[1, 0].imshow(image_label_overlay)

        ax[1, 1].imshow(img, zorder=1)
        ax[1, 2].imshow(img, zorder=1)

        plt.tight_layout()

        if save_image:
            try:
                plt.savefig(save_path + f'{ii:04d}.png', format='png')
            except FileNotFoundError:
                os.makedirs(save_path)
                plt.savefig(save_path + f'{ii:04d}.png', format='png')
        if show_image:
            plt.show()

        plt.close('all')

    return


if __name__ == '__main__':
    data, hdr = load_all_data('./test_data/')

    with mp.Pool(processes=mp.cpu_count()) as pool:
        if len(hdr) != len(data):
            out = list(tqdm(pool.imap(worker, ((dat, hdr_) for dat, hdr_ in
                            zip(data, itertools.repeat(hdr, len(data)))), chunksize=1),
                            total=len(data), desc='clustering data'))
        else:
            out = list(tqdm(pool.imap(worker, ((dat, hdr_) for dat, hdr_ in
                            zip(data, hdr)), chunksize=1),
                            total=len(data), desc='clustering data'))

    df = pd.concat(out, ignore_index=True)
    #df.to_pickle('./out_bl12_158K.pkl')

    # df = pd.read_pickle('./out.pkl')

    make_figures(df, save_image=False, show_image=True)#, save_path='./imgs/158K/')
