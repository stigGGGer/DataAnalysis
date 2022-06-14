from enum import Enum

from sklearn.cluster import AgglomerativeClustering 

class AffinityType(Enum):
    euclidean = "euclidean"
    manhattan = "manhattan"
    cosine = "cosine"
    precomputed = "precomputed"
    l1 = "l1"
    l2 = "l2"


class LinkageType(Enum):
    ward = "ward"
    average = "average"
    complete = "complete"
    single = "single"

def Agglomerative_Clustering(table,parametrs):
    clustering = AgglomerativeClustering(n_clusters=parametrs[1], affinity=parametrs[2], linkage=parametrs[0])
    Y_preds = clustering.fit_predict(table)
    return [table,Y_preds]
