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
    return [table,kmeans.labels_]
