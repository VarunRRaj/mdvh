import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

MAX_POWER=6
MAX_ITER=10000

def find_sites(hist_name, out_name, num_clusters):
    df = pd.read_csv(hist_name, delimiter=", ", engine='python')

    content = df.columns.values[-1]
    coords = df.columns.values[:-1]

    kmeans = [KMeans(n_clusters=num_clusters, random_state=0, max_iter=MAX_ITER) for _ in range(MAX_POWER)]
    PhaseSpace = np.array(df.drop([content], 1).astype(float))
    BinHeights = np.array(df[content].astype(float))

    cluster_labels = [kmeans[i].fit_predict(PhaseSpace, sample_weight=BinHeights**(i)) for i in range(MAX_POWER)]
    ratios = [max_by_min(BinHeights, labelset.astype(int)) for labelset in cluster_labels]
    best_pow = ratios.index(min(ratios))

    print("Best Power: ", best_pow)
    print("Max/Min: ", ratios[best_pow])

    sites = kmeans[best_pow].cluster_centers_
    nice_path = path_thru_points(pd.DataFrame(sites, columns=coords))
    pd.DataFrame(sites).iloc[nice_path].to_csv(out_name, index=False, header=False)

def max_by_min(zs, labels):
    num_clusters = max(labels)+1
    total_weights = np.array([0.]*num_clusters)

    for i in range(len(zs)):
        l = labels[i]
        total_weights[l] += zs[i]
    
    return max(total_weights)/min(total_weights)

def path_thru_points(point_df):
    m1 = point_df.quantile(0.4)[1]
    m2 = point_df.quantile(0.8)[1]
    t1_df = point_df[point_df.iloc[:,1] <= m1]
    t2_df = point_df[(point_df.iloc[:,1] > m1) & (point_df.iloc[:,1] <= m2)]
    t3_df = point_df[point_df.iloc[:,1] > m2]
    
    path = []
    for t_df in [t1_df, t2_df, t3_df]:
        while t_df.shape[0]:
            current_index = t_df.iloc[:,0].idxmin()
            current_point = tuple(t_df.loc[current_index,:])
            t_df = t_df.drop(current_index)
            path.append(current_index)
    return np.array(path)
