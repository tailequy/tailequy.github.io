import numpy as np
import math
import random
from utils import balance_score
class baseline:
    def __init__(self):
        #self.n_cluster = k
        self.clusters = {}
        self.cost = []
        self.__data = None
        self.__wish = None
        self.__topic = None
        self.__value = None

    def fit(self, data, wish, topic, min_cap, max_cap, protected_att, male):
        self.min_cap = min_cap
        self.max_cap = max_cap
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
        #Step 1: Assign students into groups based on their wishes
        # for each student, try to assign with their highest priority

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
        #Assign
        for i in range(self.n_students):
            for j in range(self.n_wishes):
                if flag[i] == 0:
                    if self.__wish[i][j] > 0:
                        n_student_m = self.count_non_protected_group(clusters[self.__wish[i][j] - 1])
                        n_student_f = self.count_protected_group(clusters[self.__wish[i][j] - 1])
                        if self.__data[self.protected_att][i] == self.male:
                            if (n_student_m < self.min_cap - min_f) :
                                if self.check_priority(i, j) == True:
                                    clusters[self.__wish[i][j] - 1].append(i)
                                    flag[i] = 1
                        else:
                            if (n_student_f < min_f):
                                if self.check_priority(i, j) == True:
                                    clusters[self.__wish[i][j] - 1].append(i)
                                    flag[i] = 1

        ###

        cnt = 0
        for id in range(self.n_students):
            if flag[id] == 0:
                cnt += 1

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

    def check_priority(self, index, prio):
        priority_list = []
        for i in range(self.n_students):
            if (i != index) and (self.__wish[i][prio] == self.__wish[index][prio]) and (self.__wish[index][prio]>0):
                priority_list.append(self.__topic[i][self.__wish[index][prio]-1])

        if len(priority_list) == 0:
            return True
        else:
            check = True
            for j in range(len(priority_list)):
                if self.__topic[index][self.__wish[index][prio]-1] < priority_list[j]:
                    check = False
            if check == True:
                return False
            else:
                return True
    def calculate_value(self):
        v = np.zeros((self.n_students, self.n_topics))
        for i in range(self.n_students):
            for j in range(self.n_wishes):
                v[i][self.__wish[i][j] - 1] = self.n_wishes / (j + 1)
        return v
    def check_capacity(self,clusters):

        for i in clusters.keys():
            if (len(clusters[i]) > 0) and (len(clusters[i])<self.min_cap):
                return False
        return True
    def contain_wish(self,index,wish):
        for i in range(self.n_wishes):
            if self.__wish[index][i] == wish:
                return True
        return False







