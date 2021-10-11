from scipy.sparse import csr_matrix
import numpy as np
import math
import random

class KMedoids_Knapsack:
    def __init__(self, k=2, max_iter=10, tol=0.1, start_prob=0.8, end_prob=0.99, decay_lambda=0.1):
        '''Kmedoids constructor called'''
        if start_prob < 0 or start_prob >= 1 or end_prob < 0 or end_prob >= 1 or start_prob > end_prob:
            raise ValueError('Invalid input')
        self.n_cluster = k
        self.max_iter = max_iter
        self.tol = tol
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.decay = decay_lambda
        
        self.medoids = []
        self.clusters = {}
        self.tol_reached = float('inf')
        self.current_distance = 0
        
        self.__data = None
        self.__is_csr = None
        self.__rows = 0
        self.__columns = 0
        self.cluster_distances = {}
        
        
    def fit(self, data,capacity,weights):
        self.capacity = capacity
        self.weights = weights
        self.__data = data

        #Check the data
        self.__set_data_type()

        #self.__start_algo()
        self.__initialize_medoids()
        #Assign points to clusters - apply Knapsack here
        #print("Step 2: Assign point to clusters")
        self.clusters, self.cluster_distances = self.__assignment_clusters(self.medoids)
        #Try to swap medoids and point to get a lower cost, apply knapsack in the point assignment step
        #print("Step 3: Try to swap")
        self.__update_clusters()

        #Nodes
        self.nodes = []
        for medoid in self.medoids:
            node = self.clusters[medoid]
            node.sort()
            self.nodes.append(node)
        #print("Self.nodes",self.nodes)

        # Calculate cost
        self.costs = []
        for i in self.medoids:
            self.costs.append(sum([self.__get_distance(i, j) for j in self.clusters[i]]))
        #print('Cost:',self.costs)

        #print('cluster:',self.clusters)


        return self

    def assign(self):
        """
        Assigning every point in the dataset to the closest center.

        Returns:
            mapping (list) : tuples of the form (point, center)
        """

        mapping = [(node, center) for nodes, center in zip(self.nodes, self.medoids) for node in nodes]
        map_result = sorted(mapping, key=lambda x: x[0])
        return map_result

    
    def __start_algo(self):
        self.__initialize_medoids()
        self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
        self.__update_clusters()
 
    def __update_clusters(self):
        for i in range(self.max_iter):
            cluster_dist_with_new_medoids = self.__swap_and_recalculate_clusters()
            if self.__is_new_cluster_dist_small(cluster_dist_with_new_medoids) == True:
                self.clusters, self.cluster_distances = self.__assignment_clusters(self.medoids)
            else:
                break

    def __is_new_cluster_dist_small(self, cluster_dist_with_new_medoids):
        existance_dist = self.calculate_distance_of_clusters()
        new_dist = self.calculate_distance_of_clusters(cluster_dist_with_new_medoids)
        
        if existance_dist > new_dist and (existance_dist - new_dist) > self.tol:
            self.medoids = cluster_dist_with_new_medoids.keys()
            return True
        return False
    
    def calculate_distance_of_clusters(self, cluster_dist=None):
        if cluster_dist == None:
            cluster_dist = self.cluster_distances
        dist = 0
        for medoid in cluster_dist.keys():
            dist += cluster_dist[medoid]
        return dist
        
    def __swap_and_recalculate_clusters(self):
        # http://www.math.le.ac.uk/people/ag153/homepage/KmeansKmedoids/Kmeans_Kmedoids.html
        cluster_dist = {}
        for medoid in self.medoids:
            is_shortest_medoid_found = False
            for data_index in self.clusters[medoid]:
                if data_index != medoid:
                    cluster_list = list(self.clusters[medoid])
                    cluster_list[self.clusters[medoid].index(data_index)] = medoid
                    new_distance = self.calculate_inter_cluster_distance(data_index, cluster_list)
                    if new_distance < self.cluster_distances[medoid]:
                        cluster_dist[data_index] = new_distance
                        is_shortest_medoid_found = True
                        break
            if is_shortest_medoid_found == False:
                cluster_dist[medoid] = self.cluster_distances[medoid]
        #Print
        #print("Cluster distance in swap:", cluster_dist)
        return cluster_dist
    
    def calculate_inter_cluster_distance(self, medoid, cluster_list):
        distance = 0
        for data_index in cluster_list:
            distance += self.__get_distance(medoid, data_index)
        return distance/len(cluster_list)
        
    def __calculate_clusters(self, medoids):
        clusters = {}
        cluster_distances = {}
        for medoid in medoids:
            clusters[medoid] = []
            cluster_distances[medoid] = 0
        #print to see
        #print("Number of data: self.__rows",self.__rows)

        for row in range(self.__rows):
            nearest_medoid, nearest_distance = self.__get_shortest_distance_to_mediod(row, medoids)
            cluster_distances[nearest_medoid] += nearest_distance
            #assign row to  nearest medoids
            clusters[nearest_medoid].append(row)
        
        for medoid in medoids:
            cluster_distances[medoid] /= len(clusters[medoid])
        #print("Clusters in calculate cluster step:",clusters)
        #print("Cluster distances in calculate cluster step:", cluster_distances)

        return clusters, cluster_distances

    def __assignment_clusters(self, medoids):
        clusters = {}
        cluster_distances = {}
        flag = [0]*len(self.__data)
        for medoid in medoids:
            flag[medoid] = 1

        for medoid in medoids:
            clusters[medoid] = []
            cluster_distances[medoid] = 0
            '''
            Assign points to cluster by solving by Knapsack problem
            Input: 
              + Points which are not yet assign to any cluster
              + Capacity (max of weights)
              + Weights of points
            Output: cluster assignment for cluster with medoid 'medoid'          
            '''
            index=[]
            for i in range(len(self.__data)):
                if flag[i] == 0:
                    index.append(i)
            values = []
            weights = []
            for i in range(len(index)):
                values.append(math.exp(-(1/self.decay)*self.__get_distance(medoid,index[i])))
                #values.append(1.0/(self.__get_distance(medoid, index[i])+0.000001))
                weights.append(self.weights[index[i]])
            #print(index)

            selected_points = self.knapsack_dp(values, weights, len(index), self.capacity-self.weights[medoid])
            #Update flag
            #print('Selected points:')
            #print(medoid)
            clusters[medoid].append(medoid)
            for point in selected_points:
                flag[index[point]] = 1
                clusters[medoid].append(index[point])
                cluster_distances[medoid] += self.__get_distance(medoid, index[point])
                #print(index[point])
        #Find the cluster having smallest nunber of instances and assign the rest of points to that
        min_cap = 0
        for medoid in medoids:
            if len(clusters[medoid])>min_cap:
                min_cap = len(clusters[medoid])
                pos = medoid
        #Asign points to cluster 'pos'
        for i in range(len(flag)):
            if (flag[i]==0):
                #print("Chet roi nhe, tao tim duoc may roi")
                clusters[pos].append(i)
                cluster_distances[pos] += self.__get_distance(pos, i)

        length = 0
        for medoid in medoids:
            length= length + len(clusters[medoid])
        #if (length==len(self.__data)):
            #print("Good, every points is distributed")
        #else:
        #    print("Error, some points are not")

        #for row in range(self.__rows):
        #    nearest_medoid, nearest_distance = self.__get_shortest_distance_to_mediod(row, medoids)
        #    cluster_distances[nearest_medoid] += nearest_distance
        #    # assign row to  nearest medoids
        #    clusters[nearest_medoid].append(row)

        for medoid in medoids:
            cluster_distances[medoid] /= len(clusters[medoid])
        #print("Clusters in calculate cluster step:", clusters)
        #print("Cluster distances in calculate cluster step:", cluster_distances)

        return clusters, cluster_distances
        
    def __get_shortest_distance_to_mediod(self, row_index, medoids):
        min_distance = float('inf')
        current_medoid = None
        
        for medoid in medoids:
            current_distance = self.__get_distance(medoid, row_index)
            if current_distance < min_distance:
                min_distance = current_distance
                current_medoid = medoid
        return current_medoid, min_distance

    def __initialize_medoids(self):
        '''Kmeans++ initialisation'''
        self.medoids.append(random.randint(0,self.__rows-1))
        while len(self.medoids) != self.n_cluster:
            self.medoids.append(self.__find_distant_medoid())
    
    def __find_distant_medoid(self):
        distances = []
        indices = []
        for row in range(self.__rows):
            indices.append(row)
            distances.append(self.__get_shortest_distance_to_mediod(row,self.medoids)[1])
        distances_index = np.argsort(distances)
        choosen_dist = self.__select_distant_medoid(distances_index)
        return indices[choosen_dist]
    
    def __select_distant_medoid(self, distances_index):
        start_index = round(self.start_prob*len(distances_index))
        end_index = round(self.end_prob*(len(distances_index)-1)) 
        return distances_index[random.randint(start_index, end_index)]

                           
    def __get_distance(self, x1, x2):
        a = self.__data[x1].toarray() if self.__is_csr == True else np.array(self.__data[x1])
        b = self.__data[x2].toarray() if self.__is_csr == True else np.array(self.__data[x2])
        return np.linalg.norm(a-b)
    
    def __set_data_type(self):
        '''to check whether the given input is of type "list" or "csr" '''
        if isinstance(self.__data, csr_matrix):
            self.__is_csr = True
            self.__rows = self.__data.shape[0]
            self.__columns = self.__data.shape[1]
        elif isinstance(self.__data, list):
            self.__is_csr = False
            self.__rows = len(self.__data)
            self.__columns = len(self.__data[0])
        else:
            raise ValueError('Invalid input')

    '''
    ------------------------------------------------
    Use dynamic programming (DP) to solve 0/1 knapsack problem
    Time complexity: O(nW), where n is number of items and W is capacity
    ------------------------------------------------
    knapsack_dp(values,weights,n_items,capacity,return_all=False)
    Input arguments:
      1. values: a list of numbers in either int or float, specifying the values of items
      2. weights: a list of int numbers specifying weights of items
      3. n_items: an int number indicating number of items
      4. capacity: an int number indicating the knapsack capacity
      5. return_all: whether return all info, defaulty is False (optional)
    Return:
      1. picks: a list of numbers storing the positions of selected items
      2. max_val: maximum value (optional)
    ------------------------------------------------
    '''

    def knapsack_dp(self, values, weights, n_items, capacity, return_all=False):
        self.check_inputs(values, weights, n_items, capacity)
        table = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
        keep = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)

        for i in range(1, n_items + 1):
            for w in range(0, capacity + 1):
                wi = weights[i - 1]  # weight of current item
                vi = values[i - 1]  # value of current item
                if (wi <= w) and (vi + table[i - 1, w - wi] > table[i - 1, w]):
                    table[i, w] = vi + table[i - 1, w - wi]
                    keep[i, w] = 1
                else:
                    table[i, w] = table[i - 1, w]

        picks = []
        K = capacity

        for i in range(n_items, 0, -1):
            if keep[i, K] == 1:
                picks.append(i)
                K -= weights[i - 1]

        picks.sort()
        picks = [x - 1 for x in picks]  # change to 0-index

        if return_all:
            max_val = table[n_items, capacity]
            return picks, max_val
        return picks

    def check_inputs(self, values, weights, n_items, capacity):
        # check variable type
        assert (isinstance(values, list))
        assert (isinstance(weights, list))
        assert (isinstance(n_items, int))
        assert (isinstance(capacity, int))
        # check value type
        assert (all(isinstance(val, int) or isinstance(val, float) for val in values))
        assert (all(isinstance(val, int) for val in weights))
        # check validity of value
        assert (all(val >= 0 for val in weights))
        assert (n_items > 0)
        assert (capacity > 0)