import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import glob
from astropy.io import fits
import matplotlib as mpl
from skimage import filters, segmentation, morphology
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn import cluster
from sklearn.cluster import KMeans
import multiprocessing as mp
import warnings
import os
import pandas as pd


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


def load_all_data(folder_path, N_files=None):
    """
    Load in all .fits files from a given directory
    :param folder_path:
    :return: data, hdr
    """

    hdr = []
    data = []
    for ii, file in enumerate(sorted(glob.glob(folder_path + '*.fits'))):
        if N_files is not None:
            if N_files <= ii:
                break
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
    CURRENTLY NOT USING
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
        db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
        # db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]
    except:
        db_points = []
    return db_points


def cluster_data(img, img_label, regions, SPECKLE_SIZE):
    """
    cluster data using kmeans (both location and intensity) and then cluster kmeans cluster with dbscan (density-based).
    :param img: roi image
    :param img_label: labelled image
    :param regions: regions object from skimage.measure
    :param SPECKLE_SIZE: number of pixels in single speckle
    :return: kmeans: list of kmeans objects
            kmeans_points: list of points for each kmeans object
            dbpoint: list (possibly ndarray) of final point clustering
    """
    test_points = []
    kmeans_points = []
    kmeans_all = []
    for label_index, region in enumerate(regions):
        if np.sqrt(region.area) < SPECKLE_SIZE:
            continue

        speckle_cluster = np.ma.filled(np.ma.masked_where(img_label != label_index + 1,
                                                          img_label), 0) * img  # Isolate a single connected region

        kmeans, points = cluster_single_speckle_kmeans(speckle_cluster,
                                                       SPECKLE_SIZE)  # determine number of speckles in connected region
        kmeans_points.append(points)
        kmeans_all.append(kmeans)
        cluster_centers = kmeans.cluster_centers_

        for cluster_center in cluster_centers:
            test_points.append(cluster_center)

    test_points = np.asarray(test_points)
    try:
        db = cluster.DBSCAN(eps=15, min_samples=2).fit(test_points)
        # db_points = test_points[db.core_sample_indices_]
        db_points = test_points[np.where(db.labels_ == scipy.stats.mode(db.labels_).mode)]

    except ValueError:
        db_points = []

    return kmeans_all, kmeans_points, db_points


def worker(args):
    """
    Multiprocess safe implementation of main processing. Takes raw img, applies roi, filters out non-speckles,
    labels processed image, clusters using kmeans and dbscan.
    :param args: two tuple: args[0] single raw data image, args[1] hdr
    :return: dictionary with useful parameters (see bottom of function for list)
    """
    roi = (slice(180, 235), slice(150, 270))
    origin = (270, 329)
    SPECKLE_SIZE = 3.7  # 5.8 is calculated with 10um pinhole

    iter_img = args[0]
    hdr = args[1]
    img = iter_img[roi]

    speckle_filter = filter_image_data(img)  # isolate speckles
    img_label = label_image(speckle_filter)  # label image
    regions = regionprops(img_label)  # get an array of regions with properties
    kmeans, kmeans_points, db_points = cluster_data(img, img_label, regions, SPECKLE_SIZE)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            mean = np.mean(db_points, axis=0) + [roi[0].start, roi[1].start]
            std = np.std(db_points, axis=0)

        except RuntimeWarning:
            # mean of an empty slice
            mean = [0, 0]
            std = [0, 0]

    dx = origin[0] - mean[1]
    dy = origin[1] - mean[0]
    r = (np.sqrt(dx ** 2 + dy ** 2))
    phi = np.arctan(dy / dx)

    out = {}
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
    out['mean'] = mean
    out['std'] = std
    out['r'] = r
    out['phi'] = phi
    out['speckle_size'] = SPECKLE_SIZE
    out['hdr'] = hdr

    return pd.DataFrame([out])


def make_figures(df, n_figures=-1, show_image=True, save_image=False, save_path='./imgs/'):
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
    for ii, _ in enumerate(df):
        if ii < n_figures:
            df_iter = df.loc[ii]
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))

            img = df_iter['roi_image']
            orig_img = df_iter['original_image']
            img_label = label_image(df_iter['filtered_image'])
            db_points = df_iter['db_points']
            origin = df_iter['origin']
            mean = df_iter['mean']
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

                ax[1, 0].scatter(regionprops(img_label)[label_index].centroid[1],
                                 regionprops(img_label)[label_index].centroid[0],
                                 c='g', zorder=2)

                for k, col in zip(range(len(colors)), colors):
                    my_members = klabels == k
                    ax[1, 1].plot(points[my_members, 1], points[my_members, 0], '.', color=col, markersize=8, alpha=1, zorder=2)
                    ax[1, 1].plot(cluster_centers[k, 1], cluster_centers[k, 0], '.', color=col, markersize=12,
                                  markeredgecolor='k', zorder=2)

            try:
                ax[1, 2].plot(db_points[:, 1], db_points[:, 0], '.', color='r', markersize=12, markeredgecolor='k', zorder=2)
            except TypeError:
                # This happens if db_points = []
                pass

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

            if save_image:
                try:
                    plt.savefig(save_path + f'{ii:04d}.png', format='png')
                except FileNotFoundError:
                    os.makedirs(save_path)
                    plt.savefig(save_path + f'{ii:04d}.png', format='png')
            if show_image:
                plt.show()

    return


if __name__ == '__main__':
    data, hdr = load_all_data('G:/My Drive/Data/FeGe_jumps/158K/2021 12 12/Andor DO436 CCD/', N_files=10)

    print(f'Loaded data shape: {data.shape}')
    with mp.Pool(processes=mp.cpu_count()) as pool:
        out = pool.map(worker, ((dat, hdr_) for dat, hdr_ in zip(data[:10], hdr[:10])), chunksize=1)

    df = pd.concat(out, ignore_index=True)
    df.to_pickle('./out.pkl')

    df1 = pd.read_pickle('./out.pkl')

    make_figures(df1, n_figures=3, save_image=True, show_image=True)
