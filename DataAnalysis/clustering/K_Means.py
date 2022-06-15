import numpy as np
from enum import Enum
from sklearn.cluster import KMeans

class covariance_type(Enum):
    full = "full"
    tied = "tied"
    diag = "diag"
    spherical = "spherical"
    
def K_Means(table, parametrs):
    kmeans = KMeans(n_clusters=parametrs[0], random_state=parametrs[2]).fit(table)
    #gm = GaussianMixture(n_components=parametrs[0],covariance_type = parametrs[1], random_state=parametrs[2]).fit(table)
    #gm.predict([[0, 0], [12, 3]])
    return [table,kmeans.labels_]
