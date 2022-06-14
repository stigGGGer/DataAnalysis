from enum import Enum

from sklearn.cluster import OPTICS

class Affinity(Enum):
    euclidean = "euclidean"
    precomputed = "precomputed"

def Optics(table, parametrs):
    #clustering = Affinity_Propagation(n_clusters=parametrs[1], affinity=parametrs[2], linkage=parametrs[0])
    #Y_preds = clustering.fit_predict(table)
    clustering = OPTICS(min_samples = 10 , xi = 0.05 , min_cluster_size = 0.05 )
    Y_preds = clustering.fit(table)
    return [table,Y_preds]