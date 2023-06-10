from scipy.sparse import csr_matrix
import numpy as np
import math
import random

class knapsack:
    def __init__(self):
        #self.n_cluster = k
        self.clusters = {}
        self.cost = []
        self.__data = None
        self.__wish = None  #wish of student
        self.__topic = None #cost or weight on each topic


    def fit(self,data,wish,topic,min_cap,max_cap,protected_att, male):
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.__data = data
        self.__wish = wish
        self.__topic = topic
        self.protected_att = protected_att
        self.male = male
        self.n_topics = len(topic[0])
        self.n_wishes = len(wish[0])
        self.n_students = len(self.__data)
        n_student_f = 0
        for s in range(self.n_students):
            if self.__data[self.protected_att][s] != self.male:
                n_student_f = n_student_f + 1
        self.ratio = n_student_f * 1.0 / (self.n_students - n_student_f)

        self.clusters = self.__assignment_clusters()
        return self
    def count_protected_group(self, cluster):
        n_student_f = 0
        for s in cluster:
            if self.__data[self.protected_att][s] != self.male:
                n_student_f = n_student_f + 1
        return n_student_f
    def count_non_protected_group(self, cluster):
        n_student_m = 0
        for s in cluster:
            if self.__data[self.protected_att][s] == self.male:
                n_student_m = n_student_m + 1
        return n_student_m

    def __assignment_clusters(self):

    #Step 1: Assign by knapsack
    #for id=1 to n_topic
    #    open a cluster
    #    assign students to cluster = knapsack()
    #Step 2: Turning
    ##################

    #Step 1:
        #Init
        clusters = {}
        for id in range(self.n_topics):
            clusters_key = id
            clusters[clusters_key] = []
        #Flag
        flag = [0] * len(self.__data)
        #Number of protected instance
        min_f = math.ceil((self.min_cap * self.ratio)/(1+self.ratio))  #Min
        max_f = math.ceil((self.max_cap * self.ratio)/(1+self.ratio))  #Max
        n_student_f = 0
        n_student_m = 0

        #Try to assign all students to all topics
        '''
                Assign students to group by solving by Knapsack problem
                Input: 
                  + Students who are not yet assigned to any group
                  + Capacity (number of students)
                  + Weight (=1 in case individual)
                Output: Group assignment for topic id
        '''
        for id in range(self.n_topics):
            #Find the student who are not assigned
            unassigned_student = []
            for i in range(len(self.__data)):
                if flag[i] == 0:
                    unassigned_student.append(i)
            values = []
            weights = []


            for i in range(len(unassigned_student)):
                values.append(self.__topic[unassigned_student[i]][id])
                weights.append(1) #individual student

            #Find the best students with |students| <= max_cap
            n_items = len(unassigned_student)

            if (n_items>0):
                if (self.n_students % self.min_cap == 0): # Prioritize choosing the minimum number of students for the group
                    selected_points = self.knapsack_dp(values, weights, n_items, self.min_cap)
                else:
                    selected_points = self.knapsack_dp(values, weights, n_items, self.max_cap)
                #Assign selected students to the current group id
                for point in selected_points:
                    clusters[id].append(unassigned_student[point])
                    flag[unassigned_student[point]] = 1
            #Del cluster with 0 instance
            if len(clusters[id]) == 0:
                del clusters[id]
        ####
        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1

        ####

        ####
        # Is there any students without clusters?
        # Methos 1: Try to add exist clusters
        for i in range(self.n_students):
            if flag[i] == 0:
                for k in range(self.n_topics):
                    if (k in clusters.keys()) and (len(clusters[k]) < self.max_cap) and (i not in clusters[k]):
                        n_student_m = self.count_non_protected_group(clusters[k])
                        n_student_f = self.count_protected_group(clusters[k])
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.max_cap - min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                        else:
                            if (n_student_f < min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break

        # Method 2: Create a new cluster
        remain = []
        for i in range(self.n_students):
            if flag[i] == 0:
                remain.append(i)
        if len(remain) > 0:
            # Find the cluster is the most prevalent wishes of students
            count = [0] * (self.n_topics + 1)
            for i in remain:
                for j in range(self.n_wishes):
                    count[self.__wish[i][j]] += 1
            # Assign to the cluster which is the most prevalent wishes of students
            while True:
                id = np.argmax(count)-1
                if id in clusters.keys():
                    if count[id] == 0:
                        break
                    else:
                        count[id] = 0
                else:
                    clusters[id] = []
                    for i in remain:
                        n_student_m = self.count_non_protected_group(clusters[id])
                        n_student_f = self.count_protected_group(clusters[id])
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.max_cap - min_f):
                                clusters[id].append(i)
                                flag[i] = 1
                        else:
                            if (n_student_f < min_f) :
                                clusters[id].append(i)
                                flag[i] = 1
                    break
            # Re-assign some points with the same wish to remain cluster
            while len(clusters[id]) < self.max_cap:

                len_before = len(clusters[id])
                for i in clusters.keys():
                    if i != id:
                        for j in clusters[i]:
                            if (self.contain_wish(j, id) == True) and (len(clusters[id]) < self.max_cap):
                                n_student_m = self.count_non_protected_group(clusters[id])
                                n_student_f = self.count_protected_group(clusters[id])
                                clusters[i].remove(j)
                                if self.__data[self.protected_att][j] == self.male:
                                    if (n_student_m < self.max_cap - min_f):
                                        clusters[id].append(j)
                                        flag[j] = 1
                                else:
                                    if (n_student_f < min_f) :
                                        clusters[id].append(j)
                                        flag[j] = 1

                                #clusters[id].append(j)
                len_after = len(clusters[id])
                if len_before == len_after:  # It's impossible for a solution
                    break
        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1


        #Step 2: Fine-tuning
        # If cap(group) < min_cap: #Resolve the groups containing n_items, assign students in such group to others

        n_items = 1
        # before

        while self.check_capacity(clusters) == False:

            if n_items < self.min_cap:
                for i in range(self.n_topics):
                    if i in clusters.keys():

                        if len(clusters[i]) == n_items:

                            # Resolve the group
                            for j in range(n_items):
                                flag[clusters[i][j]] = 0
                                # Assign to other groups
                                # Check other wishes
                                for k in range(self.n_wishes):

                                    if flag[clusters[i][j]] == 0:
                                        if self.__wish[clusters[i][j]][k] - 1 in clusters.keys():
                                            if (self.__wish[clusters[i][j]][k] - 1 != i) and (len(
                                                    clusters[self.__wish[clusters[i][j]][k] - 1]) < self.max_cap) and (
                                                    len(clusters[self.__wish[clusters[i][j]][k] - 1]) > 0):
                                                if clusters[i][j] not in clusters[self.__wish[clusters[i][j]][k] - 1]:
                                                    n_student_m = self.count_non_protected_group(
                                                        clusters[self.__wish[clusters[i][j]][k] - 1])
                                                    n_student_f = self.count_protected_group(
                                                        clusters[self.__wish[clusters[i][j]][k] - 1])
                                                    #clusters[self.__wish[clusters[i][j]][k] - 1].append(clusters[i][j])
                                                    #flag[clusters[i][j]] = 1

                                                    if self.__data[self.protected_att][clusters[i][j]] == self.male:
                                                        if (n_student_m < self.max_cap - max_f):
                                                            clusters[self.__wish[clusters[i][j]][k] - 1].append(
                                                                clusters[i][j])
                                                            flag[clusters[i][j]] = 1
                                                    else:
                                                        if (n_student_f < min_f):
                                                            clusters[self.__wish[clusters[i][j]][k] - 1].append(
                                                                clusters[i][j])
                                                            flag[clusters[i][j]] = 1

                                # If can not assign to other groups based on wishes, randomly assign to a group
                                if flag[clusters[i][j]] == 0:
                                    for k in range(self.n_topics):
                                        if flag[clusters[i][j]] == 0:
                                            if (k in clusters.keys()) and (k != i) and (
                                                    len(clusters[k]) < self.max_cap) and (len(clusters[k]) > 0) and (
                                                    clusters[i][j] not in clusters[k]):
                                                n_student_m = self.count_non_protected_group(clusters[k])
                                                n_student_f = self.count_protected_group(clusters[k])
                                                clusters[k].append(clusters[i][j])
                                                flag[clusters[i][j]] = 1
                                                break

                            del clusters[i]
                n_items = n_items + 1
            else:
                break

        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1

        ####

        ####
        # Is there any students without clusters?
        # Methos 1: Try to add exist clusters
        for i in range(self.n_students):
            if flag[i] == 0:
                for k in range(self.n_topics):
                    if (k in clusters.keys()) and (len(clusters[k]) < self.max_cap) and (i not in clusters[k]):
                        n_student_m = self.count_non_protected_group(clusters[k])
                        n_student_f = self.count_protected_group(clusters[k])
                        #clusters[k].append(i)
                        #flag[i] = 1
                        #break

                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.max_cap - min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                        else:
                            if (n_student_f < min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break


        # Method 2: Create a new cluster
        remain = []
        for i in range(self.n_students):
            if flag[i] == 0:
                remain.append(i)
        if len(remain) > 0:
            # Find the cluster is the most prevalent wishes of students
            count = [0] * (self.n_topics + 1)
            for i in remain:
                for j in range(self.n_wishes):
                    count[self.__wish[i][j]] += 1
            # Assign to the cluster which is the most prevalent wishes of students
            while True:
                id = np.argmax(count) - 1
                if id in clusters.keys():
                    if count[id] == 0:
                        break
                    else:
                        count[id] = 0
                else:
                    clusters[id] = []
                    for i in remain:
                        n_student_m = self.count_non_protected_group(clusters[id])
                        n_student_f = self.count_protected_group(clusters[id])
                        clusters[id].append(i)
                        flag[i] = 1
                    '''
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.min_cap - min_f):
                                clusters[id].append(i)
                                flag[i] = 1
                        else:
                            if (n_student_f < min_f):
                                clusters[id].append(i)
                                flag[i] = 1
                    '''
                    break
            # Re-assign some points with the same wish to remain cluster
            while len(clusters[id]) < self.max_cap:

                len_before = len(clusters[id])
                for i in clusters.keys():
                    if i != id:
                        for j in clusters[i]:
                            if (self.contain_wish(j, id) == True) and (len(clusters[id]) < self.max_cap):
                                n_student_m = self.count_non_protected_group(clusters[id])
                                n_student_f = self.count_protected_group(clusters[id])
                                clusters[i].remove(j)
                                clusters[id].append(j)
                                flag[j] = 1
                            '''
                                if self.__data[self.protected_att][j] == self.male:
                                    if (n_student_m < self.max_cap - min_f):
                                        clusters[id].append(j)
                                        flag[j] = 1
                                else:
                                    if (n_student_f < min_f):
                                        clusters[id].append(j)
                                        flag[j] = 1
                            '''

                                # clusters[id].append(j)
                len_after = len(clusters[id])
                if len_before == len_after:  # It's impossible for a solution
                    break
        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1


        # Is there any students without clusters?
        # Methos 1: Try to add exist clusters
        for i in range(self.n_students):
            if flag[i] == 0:
                for k in range(self.n_topics):
                    if (k in clusters.keys()) and (len(clusters[k]) < self.max_cap) and (i not in clusters[k]):
                        n_student_m = self.count_non_protected_group(clusters[k])
                        n_student_f = self.count_protected_group(clusters[k])
                        clusters[k].append(i)
                        flag[i] = 1
                        break
                    '''
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.max_cap - min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                        else:
                            if (n_student_f < min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                    '''

        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1

        # Step 2: Fine-tuning
        # If cap(group) < min_cap: #Resolve the groups containing n_items, assign students in such group to others

        n_items = 1
        # before

        while self.check_capacity(clusters) == False:

            if n_items < self.min_cap:
                for i in range(self.n_topics):
                    if i in clusters.keys():

                        if len(clusters[i]) == n_items:

                            # Resolve the group
                            for j in range(n_items):
                                flag[clusters[i][j]] = 0
                                # Assign to other groups
                                # Check other wishes
                                for k in range(self.n_wishes):

                                    if flag[clusters[i][j]] == 0:
                                        if self.__wish[clusters[i][j]][k] - 1 in clusters.keys():
                                            if (self.__wish[clusters[i][j]][k] - 1 != i) and (len(
                                                    clusters[self.__wish[clusters[i][j]][k] - 1]) < self.max_cap) and (
                                                    len(clusters[self.__wish[clusters[i][j]][k] - 1]) > 0):
                                                if clusters[i][j] not in clusters[self.__wish[clusters[i][j]][k] - 1]:
                                                    n_student_m = self.count_non_protected_group(
                                                        clusters[self.__wish[clusters[i][j]][k] - 1])
                                                    n_student_f = self.count_protected_group(
                                                        clusters[self.__wish[clusters[i][j]][k] - 1])
                                                    # clusters[self.__wish[clusters[i][j]][k] - 1].append(clusters[i][j])
                                                    # flag[clusters[i][j]] = 1

                                                    if self.__data[self.protected_att][clusters[i][j]] == self.male:
                                                        if (n_student_m < self.max_cap - max_f):
                                                            clusters[self.__wish[clusters[i][j]][k] - 1].append(
                                                                clusters[i][j])
                                                            flag[clusters[i][j]] = 1
                                                    else:
                                                        if (n_student_f < min_f):
                                                            clusters[self.__wish[clusters[i][j]][k] - 1].append(
                                                                clusters[i][j])
                                                            flag[clusters[i][j]] = 1

                                # If can not assign to other groups based on wishes, randomly assign to a group
                                if flag[clusters[i][j]] == 0:
                                    for k in range(self.n_topics):
                                        if flag[clusters[i][j]] == 0:
                                            if (k in clusters.keys()) and (k != i) and (
                                                    len(clusters[k]) < self.max_cap) and (len(clusters[k]) > 0) and (
                                                    clusters[i][j] not in clusters[k]):
                                                n_student_m = self.count_non_protected_group(clusters[k])
                                                n_student_f = self.count_protected_group(clusters[k])
                                                clusters[k].append(clusters[i][j])
                                                flag[clusters[i][j]] = 1
                                                break

                            del clusters[i]
                n_items = n_items + 1
            else:
                break

        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1

        ####

        ####
        # Is there any students without clusters?
        # Methos 1: Try to add exist clusters
        for i in range(self.n_students):
            if flag[i] == 0:
                for k in range(self.n_topics):
                    if (k in clusters.keys()) and (len(clusters[k]) < self.max_cap) and (i not in clusters[k]):
                        n_student_m = self.count_non_protected_group(clusters[k])
                        n_student_f = self.count_protected_group(clusters[k])
                        # clusters[k].append(i)
                        # flag[i] = 1
                        # break

                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.max_cap - min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                        else:
                            if (n_student_f < min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break

        # Method 2: Create a new cluster
        remain = []
        for i in range(self.n_students):
            if flag[i] == 0:
                remain.append(i)
        if len(remain) > 0:
            # Find the cluster is the most prevalent wishes of students
            count = [0] * (self.n_topics + 1)
            for i in remain:
                for j in range(self.n_wishes):
                    count[self.__wish[i][j]] += 1
            # Assign to the cluster which is the most prevalent wishes of students
            while True:
                id = np.argmax(count) - 1
                if id in clusters.keys():
                    if count[id] == 0:
                        break
                    else:
                        count[id] = 0
                else:
                    clusters[id] = []
                    for i in remain:
                        n_student_m = self.count_non_protected_group(clusters[id])
                        n_student_f = self.count_protected_group(clusters[id])
                        clusters[id].append(i)
                        flag[i] = 1
                    '''
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.min_cap - min_f):
                                clusters[id].append(i)
                                flag[i] = 1
                        else:
                            if (n_student_f < min_f):
                                clusters[id].append(i)
                                flag[i] = 1
                    '''
                    break
            # Re-assign some points with the same wish to remain cluster
            while len(clusters[id]) < self.max_cap:

                len_before = len(clusters[id])
                for i in clusters.keys():
                    if i != id:
                        for j in clusters[i]:
                            if (self.contain_wish(j, id) == True) and (len(clusters[id]) < self.max_cap):
                                n_student_m = self.count_non_protected_group(clusters[id])
                                n_student_f = self.count_protected_group(clusters[id])
                                clusters[i].remove(j)
                                clusters[id].append(j)
                                flag[j] = 1
                            '''
                                if self.__data[self.protected_att][j] == self.male:
                                    if (n_student_m < self.max_cap - min_f):
                                        clusters[id].append(j)
                                        flag[j] = 1
                                else:
                                    if (n_student_f < min_f):
                                        clusters[id].append(j)
                                        flag[j] = 1
                            '''

                            # clusters[id].append(j)
                len_after = len(clusters[id])
                if len_before == len_after:  # It's impossible for a solution
                    break
        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1


        # Is there any students without clusters?
        # Methos 1: Try to add exist clusters
        for i in range(self.n_students):
            if flag[i] == 0:
                for k in range(self.n_topics):
                    if (k in clusters.keys()) and (len(clusters[k]) < self.max_cap) and (i not in clusters[k]):
                        n_student_m = self.count_non_protected_group(clusters[k])
                        n_student_f = self.count_protected_group(clusters[k])
                        clusters[k].append(i)
                        flag[i] = 1
                        break
                    '''
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.max_cap - min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                        else:
                            if (n_student_f < min_f):
                                clusters[k].append(i)
                                flag[i] = 1
                                break
                    '''

        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1

        #Remove empty clusters
        key_remove = []
        for key in clusters.keys():
            if (len(clusters[key])==0):
                key_remove.append(key)
        for key in key_remove:
            del clusters[key]


        return clusters

    def calculate_value(self):
        v = np.zeros((self.n_students, self.n_topics))
        for i in range(self.n_students):
            for j in range(self.n_wishes):
                v[i][self.__wish[i][j] - 1] = self.n_wishes / (j + 1)
        return v
    def check_capacity(self,clusters):

        for i in range(self.n_topics):
            if i in clusters.keys():
                if (len(clusters[i]) > 0) and (len(clusters[i])< self.min_cap):
                    return False
        return True


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

    def contain_wish(self,index,wish):
        for i in range(self.n_wishes):
            if self.__wish[index][i] == wish:
                return True
        return False