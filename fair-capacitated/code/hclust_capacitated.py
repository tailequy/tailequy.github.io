import sys
import math
import os
import heapq
import itertools
from utils import distance

class Hierarchical_Clustering(object):
    def __init__(self, k=5):
        #self.input_file_name = ipt_data
        self.k = k
        self.dataset = None
        self.dataset_size = 0
        self.dimension = 0
        self.heap = []
        self.clusters = []
        self.centers = []
        #self.gold_standard = {}

    def compute_pairwise_distance(self, dataset):
        result = []
        dataset_size = len(dataset)
        for i in range(dataset_size - 1):  # ignore last i
            for j in range(i + 1, dataset_size):  # ignore duplication
                dist = distance(dataset[i]["data"], dataset[j]["data"])

                # duplicate dist, need to be remove, and there is no difference to use tuple only
                # leave second dist here is to take up a position for tie selection
                result.append((dist, [dist, [[i], [j]]]))

        return result

    def build_priority_queue(self, distance_list):
        heapq.heapify(distance_list)
        self.heap = distance_list
        return self.heap

    def compute_centroid_two_clusters(self, current_clusters, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0] * dim
        for index in data_points_index:
            dim_data = current_clusters[str(index)]["centroid"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def compute_centroid(self, dataset, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0] * dim
        for idx in data_points_index:
            dim_data = dataset[idx]["data"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def valid_heap_node(self, heap_node, old_clusters):
        pair_dist = heap_node[0]
        pair_data = heap_node[1]
        for old_cluster in old_clusters:
            if old_cluster in pair_data:
                return False
        return True

    def add_heap_entry(self, heap, new_cluster, current_clusters,dataset,linkage):
        for ex_cluster in current_clusters.values():
            new_heap_entry = []

            #dist = distance(ex_cluster["centroid"], new_cluster["centroid"])


            #Print to see
            old_index = ex_cluster["elements"]
            new_index = new_cluster["elements"]
            #print("Old index",old_index)
            #print("New index", new_index)
            #if len(old_index)==0:
            #    print("Old index is empty")
            #if len(new_index)==0:
            #    print("New index is empty")


            #if len(old_index)>0 and len(new_index)>0:
            #    #Calculate distance
            distamce_array = []
            for i in range(len(old_index)):
                for j in range(len(new_index)):
                    distamce_array.append(distance(dataset[old_index[i]]["data"], dataset[new_index[j]]["data"]))
            #average
            if linkage=='average':
                dist = sum(distamce_array)/len(distamce_array)
            elif linkage=='max':
                #max
                dist = max(distamce_array)
            elif linkage == 'min':
                #min
                dist = min(distamce_array)
            elif linkage =='centroid':
                #centroid
                dist = distance(ex_cluster["centroid"], new_cluster["centroid"])

            #print(dist)

            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))

        #for ex_cluster in current_clusters.values():
        #    new_heap_entry = []
        #    dist = distance(ex_cluster["centroid"], new_cluster["centroid"])
        #    new_heap_entry.append(dist)
        #    new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
        #    heapq.heappush(heap, (dist, new_heap_entry))

    def fit(self, data, capacity, weights,linkage):
        self.capacity = capacity
        self.weights = weights

        dataset = []
        clusters = {}
        for id in range(len(data)):
            line = data[id]

            data_temp = {}
            data_temp.setdefault("id", id)  # duplicate
            data_temp.setdefault("data", line)
            dataset.append(data_temp)

            clusters_key = str([id])
            clusters.setdefault(clusters_key, {})
            clusters[clusters_key].setdefault("centroid", line)
            clusters[clusters_key].setdefault("elements", [id])
        self.dataset = dataset
        self.clusters = clusters
        self.dataset_size = len(self.dataset)
        self.dimension = len(self.dataset[0]["data"])
        ### Main Hierarchical clustering
        #dataset = self.dataset
        # print(self.clusters)
        current_clusters = self.clusters
        old_clusters = []
        heap = self.compute_pairwise_distance(dataset)
        heap = self.build_priority_queue(heap)
        try:
            while len(current_clusters) > self.k:
                #print("Len of current clusters",len(current_clusters))

                if len(heap) == 0:
                    return current_clusters
                # print('Len of heap: ', len(heap))
                dist, min_item = heapq.heappop(heap)
                # pair_dist = min_item[0]
                pair_data = min_item[1]
                ######
                sumofweight = 0
                length0 = len(pair_data[0])
                length1 = len(pair_data[1])
                # print('Length 0:',length0, ' length 1:', length1)
                for i in range(length0):
                    sumofweight = sumofweight + self.weights[pair_data[0][i]]
                for i in range(length1):
                    sumofweight = sumofweight + self.weights[pair_data[1][i]]
                # print('Length 0:', length0, ' length 1:', length1, 'sum of weight', sumofweight)
                if sumofweight > self.capacity:
                #print('Sum of weights: ', sumofweight, 'continue')
                    continue

                # judge if include old cluster
                if not self.valid_heap_node(min_item, old_clusters):
                    continue

                new_cluster = {}
                new_cluster_elements = sum(pair_data, [])
                new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
                new_cluster_elements.sort()
                new_cluster.setdefault("centroid", new_cluster_cendroid)
                new_cluster.setdefault("elements", new_cluster_elements)

                for pair_item in pair_data:
                    old_clusters.append(pair_item)
                    del current_clusters[str(pair_item)]
                self.add_heap_entry(heap, new_cluster, current_clusters,dataset,linkage)
                current_clusters[str(new_cluster_elements)] = new_cluster

                #print(current_clusters)
        except ValueError:
            print("Runtime Error")
        clusters_results = current_clusters.values()
        #print(clusters_results)
        #Print
        #print("Dataset:", dataset)
        #find the center (medoid)
        for medoid in clusters_results:
            #print(medoid["centroid"])
            mindis = 1000000000.0
            for id in medoid["elements"]:
                if distance(dataset[id]["data"],medoid["centroid"]) < mindis:
                    mindis = distance(dataset[id]["data"],medoid["centroid"])
                    current_medoid = id
            self.centers.append(current_medoid)
        #Print
        #print("Center: ", self.centers)

        self.nodes = []
        for cluster in clusters_results:
            cluster["elements"].sort()
            self.nodes.append(cluster["elements"])
        #
        ###
        self.costs = []
        for i in range(len(self.centers)):
            self.costs.append(max([distance(self.dataset[self.centers[i]]["data"], self.dataset[j]["data"]) for j in self.nodes[i]]))

        # Print clusters
        i = 0
        count = 0
        for cluster in clusters_results:
            cluster["elements"].sort()
            #print('Cluster ', i, ' len: ', len(cluster["elements"]))
            count = count + len(cluster["elements"])
            #print(cluster["elements"])
            i = i + 1
        #print('#instance:', count)
        #
        return

    def assign(self):
        """
        Assigning every point in the dataset to the closest center.

        Returns:
            mapping (list) : tuples of the form (point, center)
        """
        mapping = [(node, center) for nodes, center in zip(self.nodes, self.centers) for node in nodes]
        map_result = sorted(mapping, key=lambda x: x[0])
        return map_result



