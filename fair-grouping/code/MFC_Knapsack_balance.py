from scipy.sparse import csr_matrix
import numpy as np
import math
import random

class MFC_knapsack:
    def __init__(self):
        #self.n_cluster = k
        self.clusters = {}
        self.cost = []
        self.__data = None
        self.__wish = None  #wish of student
        self.__topic = None #cost or weight on each topic


    def fit(self,data,wish,topic,min_cap,max_cap,theta,protected_att, male):
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.theta = theta
        self.protected_att = protected_att
        self.male = male
        self.__data = data
        self.__wish = wish
        self.__topic = topic
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
    #Step 2: Turn
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
                Assign students to group by solving by MFC_Knapsack problem
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


            n_students_m = 0
            for i in range(len(unassigned_student)):
                values.append(self.__topic[unassigned_student[i]][id])
                weights.append(1) #individual student

                if (self.__data[self.protected_att][unassigned_student[i]] == self.male):
                    n_students_m = n_students_m + 1

            #Find the best students with min_cap <= |students| <= max_cap
            n_items = len(unassigned_student)
            n_students_f = n_items - n_students_m


            if (n_items>0):
                #Parameters: values, weights, n_students, n_students_f, n_students_m, capacity_min, capacity_max, theta, return_all
                if (self.n_students % self.min_cap == 0): # Prioritize choosing the minimum number of students for the group
                    selected_points = self.mfc_knapsack_dp(values, weights, n_items, n_students_f, n_students_m, self.min_cap, self.min_cap,self.theta)
                else:
                    selected_points = self.mfc_knapsack_dp(values, weights, n_items, n_students_f, n_students_m, self.min_cap, self.max_cap,self.theta)

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

        #Remove duplicate values
        for key in clusters.keys():
            clusters[key] = list(dict.fromkeys(clusters[key]))

        for id in clusters.keys():
            for value in clusters[id]:
                for id2 in clusters.keys():
                    for value2 in clusters[id2]:
                        if (id != id2) and (value == value2):
                            clusters[id].remove(value)

        return clusters


    def calculate_value(self):
        v = np.zeros((self.n_students, self.n_topics))
        for i in range(self.n_students):
            for j in range(self.n_wishes):
                v[i][self.__wish[i][j] - 1] = self.n_wishes / (j + 1)
        return v
    def check_assign(self,clusters,value):
        for key in clusters.keys():
            for v in clusters[key]:
                if value == v:

                    return False
        return True
    def check_capacity(self,clusters):

        for i in range(self.n_topics):
            if i in clusters.keys():
                if (len(clusters[i]) > 0) and (len(clusters[i])< self.min_cap):
                    return False
        return True

    '''
        ------------------------------------------------
        Use dynamic programming (DP) to solve MFC knapsack problem
        Time complexity: O(nw), where n is number of students and w is capacity
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
    def mfc_knapsack_dp(self, values, weights, n_students, n_students_f, n_students_m, capacity_min, capacity_max, theta, return_all=False):
        self.mfc_check_inputs(values, weights, n_students, n_students_f,n_students_m, capacity_min,capacity_max)
        p = 2
        table_A = np.zeros((p, n_students + 1, capacity_max + 1), dtype=np.float32) #Max total values (cost) of the first s student with capacity w, group p
        keep_A = np.zeros((p, n_students + 1, capacity_max + 1), dtype=np.float32)  #Keep track of chosen students in table_A

        table_B = np.zeros((p, capacity_max + 1), dtype=np.float32) #Max total cost of group p, capacity w
        keep_B = np.zeros((p, capacity_max + 1), dtype=np.float32)

        #Table A
        for p in range(0, 2):
            for s in range(1, n_students + 1):
                for w in range(0, capacity_max + 1):
                    ws = weights[s - 1]  # weight of current student
                    vs = values[s - 1]  # value (cost) of current student
                    if (ws <= w) and (vs + table_A[p, s - 1, w - ws] > table_A[p,s - 1, w]):
                        table_A[p, s, w] = vs + table_A[p, s - 1, w - ws]
                        keep_A[p, s, w] = 1
                    else:
                        table_A[p,s, w] = table_A[p,s - 1, w]

        #Table B

        p0l = math.ceil(capacity_min/((1+theta)/theta))
        p0u = math.ceil(capacity_max/((1+theta)/theta)) + 1 #Add one more
        for w in range(p0l,p0u+1):
            max_B = 0
            for s in range(0,n_students_f+1):
                if (max_B < table_A[0,s,w]):
                    max_B = table_A[0,s,w]
            if max_B >0:
                table_B[0,w] = max_B
                keep_B[0,w]=1

        p1l = capacity_min - p0l
        p1u = capacity_max - p0u + 1
        for w in range(capacity_min,capacity_max):
            max_B = 0
            keep_w = 0
            for s in range(0,n_students_m):
                for w_f in range(p0l,p0u+1):
                    if p1l <= (w-w_f) and (w-w_f) <= p1u:
                        if (w!=w_f):
                            if (w_f/(w-w_f)>=theta):
                                if max_B < table_B[0,w-w_f] + table_A[1,s,w-w_f]:
                                    max_B = table_B[0,w-w_f] + table_A[1,s,w-w_f]
                                    keep_w = w-w_f  #weight of male

                if max_B>0:
                    table_B[1,w] = max_B
                    keep_B[1,keep_w]  = 1
                    keep_B[1,w] = w_f #weight of female
                else:
                    table_B[1,w] = table_B[0,w-w_f]

        avg = sum(values)/((capacity_max+capacity_min)/2.0)
        min_dif = abs(table_B[1,capacity_min] - avg)
        w_best = capacity_min
        for w in range(capacity_min,capacity_max+1):
            if min_dif > abs(table_B[1,w]-avg):
                min_dif = abs(table_B[1,w]-avg)
                w_best = w
        max_val = table_B[1, w_best]
        #Tracking
        picks = []
        w_best_f = keep_B[1, w_best]
        w_best_m = w_best - keep_B[1, w_best]
        K = int(w_best_f)
        for s in range(n_students_f, 0, -1):

            if (keep_A[0, s, K] == 1):
                picks.append(s)
                K -= weights[s-1]
        K = int(w_best_m)
        for s in range(n_students_m, 0, -1):
            if (keep_A[1, s, K] == 1):
                picks.append(s)
                K -= weights[s - 1]
        picks.sort()
        picks = [x - 1 for x in picks]  # change to 0-index

        if return_all:
            return picks, max_val
        return picks

    def mfc_check_inputs(self, values, weights, n_students, n_students_f,n_students_m,capacity_min, capacity_max):
        # check variable type
        assert (isinstance(values, list))
        assert (isinstance(weights, list))
        assert (isinstance(n_students, int))
        assert (isinstance(n_students_f, int))
        assert (isinstance(n_students_m, int))
        assert (isinstance(capacity_min, int))
        assert (isinstance(capacity_max, int))
        # check value type
        assert (all(isinstance(val, int) or isinstance(val, float) for val in values))
        assert (all(isinstance(val, int) for val in weights))
        # check validity of value
        assert (all(val >= 0 for val in weights))
        assert (n_students > 0)
        #assert (n_students_f > 0)
        #assert (n_students_m > 0)
        assert (capacity_min > 0)
        assert (capacity_max > 0)

    def contain_wish(self,index,wish):
        for i in range(self.n_wishes):
            if self.__wish[index][i] == wish:
                return True
        return False