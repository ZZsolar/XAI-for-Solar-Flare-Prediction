# This is the main programme for calculating Polarity Inversion Line with built wheel.
# sunpy, sklearn are the two main packages.
"""
Brief introduction: This is the main programme for locating the area of polarity inversion line of a series of active regions with multiple cores.

file structure:
             > PIL (main folder)
               Locate_PIL.py (This code)
               > data
               > res
                 > Br
                 > bitmap_positive
                 > bitmap_negative
                 > map_pil
                 > coord_pil

input: Br.fits files from JSOC.

output: .txt file containing coordinates of the polarity inversion line in 'coord_pil' folder
        Maps of Br, bitmap_positive, bitmap_negative, map_pil in the rest of the folders in 'res'

principle: 1. Read the Br.fits files and get the map.data, which is supposed to be a two-dimension list
           2. Set sigma = 100G (Hoekesema et al., 2014), we use 2*sigma as the threshold to get bitmap_positive and bitmap_negative.
              In bitmap_positive, any pixel with value mt 200G is set to be 1, the rest are 0.
              In bitmap_negative, any pixel with value lt -200G is set to be -1, the rest are 0.
           3. We use DBSCAN algorithm on the bitmaps, which is for removing small clusters that may influence the result.
           4. We set up a Gaussian kernel with width as 10 pixels(width is adjustable).
              Then convolve the two bitmaps with the Gaussian kernel.
           5. Multiply the two bitmaps that is Gaussian-kernel processed to get the needed PIL-Inversion_Line map.

modified by: Z. Zheng
==================================================================================================================
"""

import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter

def get_coordinates(B_map_data, threshold, sign):
    """
    :param map: Directly get from the Br.fits.
    :param threshold: According to Hoeksema, this value is often chosen to be 200.
    :param sign: string, 'positive' or 'negative'
    :return: a list of coordinates, which show the position of pixels that meet the threshold.
    """
    coordinates = []
    if sign == 'positive':
        for i in range(B_map_data.shape[0]):
            for j in range(B_map_data.shape[1]):
                if B_map_data[i][j] > threshold:
                    coordinates.append([i,j])
    else:
        for i in range(B_map_data.shape[0]):
            for j in range(B_map_data.shape[1]):
                if B_map_data[i][j] < -threshold:
                    coordinates.append([i,j])
    return coordinates

def read_fits(fitsfile):
        from astropy.io import fits
        hdu = fits.open(fitsfile)
        try:
            data = hdu[1].data
            header = hdu[1].header
    
            return data, header
        except:
            data = hdu[0].data
            header = hdu[0].header
    
            return data, header

def find_biggest_cluster(cluster, n):
    """
    From DBSCAN result, return the top n largest clusters.
    :param cluster: fitted DBSCAN object
    :param n: number of clusters to select
    :return: list of [cluster_index, point_count]
    """
    labels = cluster.labels_.tolist()
    valid_labels = [lab for lab in labels if lab != -1]
    if not valid_labels:
        return []

    counts = []
    for i in range(max(valid_labels) + 1):
        counts.append([i, valid_labels.count(i)])

    counts.sort(key=lambda x: x[1], reverse=True)

    return counts[:min(n, len(counts))]


def coordinates_of_clusters(cluster_dbscan, cluster_fbc, n_cluster):
    """
    This function will return the coordinates of the clusters selected.
    Handles the case when the number of clusters found is less than n_cluster.
    
    :param cluster_dbscan: cluster variables directly from DBSCAN
    :param cluster_fbc: cluster variables from find_biggest_cluster(fbc)
    :param n_cluster: number of clusters requested
    :return: a list of lists, each sublist contains coordinates of a chosen cluster
    """
    coordinates_nc = [[] for _ in range(n_cluster)]
    labels = [x for x in cluster_dbscan.labels_ if x != -1]
    actual_clusters = min(len(cluster_fbc), n_cluster)
    for i in range(len(labels)):
        for j in range(actual_clusters):
            if labels[i] == cluster_fbc[j][0]:
                coordinates_nc[j].append(cluster_dbscan.components_[i].tolist())
    return coordinates_nc


def coordinates2map(map_size, coordinates, sign):
    """
    :param map_size: Size of the map data, list
    :param coordinates: The coordinates that will be set non-zero.
    :param sign: 'positive' or 'negative', wihch decides the sign of the bitmap.
    :return: a bitmap data in which the chosen coordinates are non-zero which others are zero.
    """
    if sign == 'positive':
        a = 1
    else:
        a = -1

    map = np.zeros(map_size).tolist()
    for i in range(len(coordinates)):
        for j in range(len(coordinates[i])):
            map[int(coordinates[i][j][0])][int(coordinates[i][j][1])] = a

    return map

def map_PIL(bitmapp, bitmapn):
    """
    using "dilate" algorithm to expand the two kinds of bitmaps to get the pil area.
    :param bitmapp: the positive bitmap
    :param bitmapn: the negative bitmap
    :return: a list containing the map data of the PIL and the coordinates of the PIL.
    """
    size = [len(bitmapp), len(bitmapp[0])]
    map = np.zeros(size)
    coordinates = []
    for i in range(size[0]):
        for j in range(size[1]):
            if bitmapp[i][j] != 0 and bitmapn[i][j] != 0:
                map[i][j] = bitmapp[i][j]
                coordinates.append([i, j])
    return map, coordinates

def change_mapdata(br, new_mapdata):
    shape = br.data.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            br.data[i][j] = new_mapdata[i][j]

    return br

def main_run(input):
    
    try:
        harpnum, file_name = input
        data_path = f'/data1/data_zz/data_sharp_cea/sharp_{harpnum:05d}/'
        
        br_mapdata, _ = read_fits(data_path + file_name + ".Br.fits")
        # Threshold 200 G for strong polarity (Hoeksema et al.)
        coordinates_posi = get_coordinates(br_mapdata, 200, 'positive')
        coordinates_nega = get_coordinates(br_mapdata, 200, 'negative')

        cluster_positive = DBSCAN(eps=1., min_samples=2).fit(coordinates_posi)
        cluster_negative = DBSCAN(eps=1., min_samples=2).fit(coordinates_nega)
        n_cluster = 5  # number of positive/negative polarity clusters to keep
        clusterp = find_biggest_cluster(cluster_positive, n_cluster)
        clustern = find_biggest_cluster(cluster_negative, n_cluster)
        coordinates_ncp = coordinates_of_clusters(cluster_positive, clusterp, n_cluster)
        coordinates_ncn = coordinates_of_clusters(cluster_negative, clustern, n_cluster)
        size_map = [len(br_mapdata), len(br_mapdata[0])]
        bitmap_positive = coordinates2map(size_map, coordinates_ncp, 'positive')
        bitmap_negative = coordinates2map(size_map, coordinates_ncn, 'negative')

        bitmapp_gs = gaussian_filter(bitmap_positive, sigma = 2)
        bitmapn_gs = gaussian_filter(bitmap_negative, sigma = 2)
        map_pil, coordinates_pil = map_PIL(bitmapp_gs, bitmapn_gs)
        img_pil = ((map_pil > 0) * 1).astype(np.uint8)

        np.save(f'/data_work/mask_data/pil/{file_name}.pil.npy', img_pil)
        print(f"{file_name} done")
        return None
    except Exception as e:
        print(f"Error processing event: {e}")
        return None

from concurrent.futures import ProcessPoolExecutor

def main():
    
    df = pd.read_csv("../data/label.csv")
    
    with ProcessPoolExecutor(max_workers=32) as executor:

        futures = []
        for _, row in df.iterrows():
            harpnum = row['HARPNUM']
            file_name = row['file_name']
            futures.append(executor.submit(main_run, [harpnum, file_name]))
        
        for future in futures:
            future.result()  

    print("all tasks done")
    return None

if __name__ == "__main__":
    main()

