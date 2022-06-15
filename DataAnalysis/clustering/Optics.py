from enum import Enum

from sklearn.cluster import OPTICS


def Optics(table, parametrs):
    clustering = OPTICS(min_samples = parametrs[0] , xi = parametrs[1] , min_cluster_size = parametrs[2]).fit(table)    
    return [table,clustering.labels_]