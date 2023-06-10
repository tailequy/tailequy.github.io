#Import library
import numpy as np
import pandas as pd
import random
from baseline_balance import baseline
from utils import distance, fairness_calculation, generate_data, f_cost
import time
import math
from Knapsack_balance import knapsack
from MFC_Knapsack_balance import MFC_knapsack

def experiment_baseline_old(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap,alpha, beta):
    name = []
    for i in range(n_wishes):
        name.append('Pri' + str(i + 1))
    wish_df = pd.DataFrame(columns=name)

    for i in range(n_wishes):
        column_name = 'Pri' + str(i + 1)
        wish_df[column_name] = data[column_name]
    name = []
    for i in range(n_topic):
        name.append('T' + str(i + 1))
    topic_df = pd.DataFrame(columns=name)
    for i in range(n_topic):
        column_name = 'T' + str(i + 1)
        topic_df[column_name] = data[column_name]

    n_students = len(wish_df)
    wish_values = wish_df.values

    v = np.zeros((n_students, n_topic))
    for i in range(n_students):
        for j in range(n_wishes):
            v[i][wish_values[i][j] - 1] = n_wishes / (j + 1)
    
    # Normalize
    v_max = np.max(v,axis =1)
    for i in range(n_students):
        for j in range(n_topic):
            v[i][j] = v[i][j] / v_max[i]
    
    w = topic_df.values * 1.0
    
    w_max = np.max(w, axis=0)
    # Normalize
    for i in range(n_students):
        for j in range(n_topic):
            w[i][j] = (w[i][j] * 1.0) / (w_max[j] * 1.0)
    
    # cost = f(v,w)
    cost = f_cost(v, w,alpha,beta)
    # Model
    base = baseline()
    base.fit(data, wish_df.values, cost, min_cap, min_cap + diff_cap)
    count_clusters = 0
    for i in range(n_topic):
        if i in base.clusters.keys():
            count_clusters = count_clusters + len(base.clusters[i])


    balance, capacity, nash, satisfied, tracking = fairness_calculation(data, wish_df.values, cost, base.clusters, protected_att,
                                                              male, n_wishes)

    return count_clusters, len(base.clusters), balance, capacity, nash, satisfied, tracking

def run_experiment_baseline_old(data, n_wishes, n_topic, max_of_min_cap, diff_cap = 1, protected_att = 'sex',male = 'M',alpha=1, beta=1, verbose = True):
    curr_min_cap = []
    curr_nash = []
    curr_satisfied = []
    curr_balances = []
    curr_capacities = []
    curr_n_clusters = []
    curr_count_instances = []
    curr_tracking = []
    for min_cap in range(2, max_of_min_cap+1, 1):
        start_time = time.time()
        count_instances, n_clusters, balance, capacity, nash, satisfied, tracking = experiment_baseline(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap,alpha,beta)
        curr_min_cap.append(min_cap)
        curr_nash.append(np.prod(nash))

        curr_satisfied.append(satisfied*1.0/len(data)*100)
        curr_balances.append(balance)
        curr_capacities.append(capacity)
        curr_n_clusters.append(n_clusters)
        curr_count_instances.append(count_instances)
        curr_tracking.append(tracking)
        if verbose:
            print("Time taken for min of capacity %d : %.3f seconds." % (min_cap, time.time() - start_time))
    return curr_min_cap,curr_nash, curr_satisfied, curr_balances, curr_capacities, curr_n_clusters, curr_count_instances, curr_tracking


def experiment_baseline(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap, alpha, beta):
    name = []
    for i in range(n_wishes):
        name.append('Pri' + str(i + 1))
    wish_df = pd.DataFrame(columns=name)

    for i in range(n_wishes):
        column_name = 'Pri' + str(i + 1)
        wish_df[column_name] = data[column_name]
    name = []
    for i in range(n_topic):
        name.append('T' + str(i + 1))
    topic_df = pd.DataFrame(columns=name)
    for i in range(n_topic):
        column_name = 'T' + str(i + 1)
        topic_df[column_name] = data[column_name]

    n_students = len(wish_df)
    wish_values = wish_df.values

    v = np.zeros((n_students, n_topic))
    for i in range(n_students):
        for j in range(n_wishes):
            v[i][wish_values[i][j] - 1] = n_wishes / (j + 1)

    # Normalize
    v_max = np.max(v, axis=1)
    for i in range(n_students):
        for j in range(n_topic):
            v[i][j] = v[i][j] / v_max[i]

    w = topic_df.values * 1.0

    w_max = np.max(w, axis=0)
    # Normalize
    for i in range(n_students):
        for j in range(n_topic):
            w[i][j] = (w[i][j] * 1.0) / (w_max[j] * 1.0)

    # cost = f(v,w)
    cost = f_cost(v, w, alpha, beta)
    #Save to file
    #save_df = pd.DataFrame(cost)
    #save_df.to_csv("Student_por_welfare.csv",index=False)
    # Model
    base = baseline()
    base.fit(data, wish_df.values, cost, min_cap, min_cap + diff_cap, protected_att,male)
    count_clusters = 0
    for i in range(n_topic):
        if i in base.clusters.keys():
            count_clusters = count_clusters + len(base.clusters[i])

    balance, capacity, nash, satisfied, tracking = fairness_calculation(data, wish_df.values, cost, base.clusters,
                                                                        protected_att,
                                                                        male, n_wishes)

    return count_clusters, len(base.clusters), balance, capacity, nash, satisfied, tracking


def run_experiment_baseline(data, n_wishes, n_topic, min_of_min_cap, max_of_min_cap, diff_cap=1, protected_att='sex', male='M',
                                 alpha=1, beta=1, verbose=True):
    curr_min_cap = []
    curr_nash = []
    curr_satisfied = []
    curr_balances = []
    curr_capacities = []
    curr_n_clusters = []
    curr_count_instances = []
    curr_tracking = []
    for min_cap in range(min_of_min_cap, max_of_min_cap + 1, 1):
        start_time = time.time()
        count_instances, n_clusters, balance, capacity, nash, satisfied, tracking = experiment_baseline(data,
                                                                                                        protected_att,
                                                                                                        male, n_wishes,
                                                                                                        n_topic,
                                                                                                        min_cap,
                                                                                                        diff_cap,
                                                                                                        alpha,beta)
        curr_min_cap.append(min_cap)
        for ii in range(0, len(nash)):
            if math.isnan(nash[ii]):
                nash[ii]=1.0

        curr_nash.append(np.prod(nash))
        curr_satisfied.append(satisfied * 1.0 / len(data)*100)
        curr_balances.append(balance)
        curr_capacities.append(capacity)
        curr_n_clusters.append(n_clusters)
        curr_count_instances.append(count_instances)
        curr_tracking.append(tracking)
        if verbose:
            print("Time taken for min of capacity %d : %.3f seconds." % (min_cap, time.time() - start_time))
    return curr_min_cap, curr_nash, curr_satisfied, curr_balances, curr_capacities, curr_n_clusters, curr_count_instances, curr_tracking


def experiment_knapsack(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap,alpha, beta):
    name = []
    for i in range(n_wishes):
        name.append('Pri' + str(i + 1))
    wish_df = pd.DataFrame(columns=name)

    for i in range(n_wishes):
        column_name = 'Pri' + str(i + 1)
        wish_df[column_name] = data[column_name]
    name = []
    for i in range(n_topic):
        name.append('T' + str(i + 1))
    topic_df = pd.DataFrame(columns=name)
    for i in range(n_topic):
        column_name = 'T' + str(i + 1)
        topic_df[column_name] = data[column_name]

    n_students = len(wish_df)
    wish_values = wish_df.values
    v = np.zeros((len(wish_values), n_topic))
    for i in range(n_students):
        for j in range(n_wishes):
            v[i][wish_values[i][j] - 1] = n_wishes / (j + 1)
    # Normalize
    v_max = np.max(v,axis =1)
    for i in range(n_students):
        for j in range(n_topic):
            v[i][j] = v[i][j] / v_max[i]

    w = topic_df.values * 1.0
    w_max = np.max(w, axis=0)
    # Normalize
    for i in range(n_students):
        for j in range(n_topic):
            w[i][j] = (w[i][j] * 1.0) / (w_max[j] * 1.0)
    # cost = f(v,w)
    cost = f_cost(v, w,alpha,beta)
    # Model
    Knapsack = knapsack()
    Knapsack.fit(data, wish_df.values, cost, min_cap, min_cap + diff_cap,protected_att,male)
    count_clusters = 0
    for i in range(n_topic):
        if i in Knapsack.clusters.keys():
            count_clusters = count_clusters + len(Knapsack.clusters[i])



    balance, capacity, nash, satisfied, tracking = fairness_calculation(data, wish_df.values, cost, Knapsack.clusters, protected_att,
                                                              male, n_wishes)

    return count_clusters, len(Knapsack.clusters), balance, capacity, nash, satisfied,tracking

def run_experiment_knapsack(data, n_wishes, n_topic, min_of_min_cap,max_of_min_cap, diff_cap = 1, protected_att = 'sex',male = 'M', alpha = 1, beta = 1, verbose = True):
    curr_min_cap = []
    curr_nash = []
    curr_satisfied = []
    curr_balances = []
    curr_capacities = []
    curr_n_clusters = []
    curr_count_instances = []
    curr_tracking = []
    #diff_cap = 2
    for min_cap in range(min_of_min_cap, max_of_min_cap+1, 1):
        start_time = time.time()
        count_instances, n_clusters, balance, capacity, nash, satisfied, tracking  = experiment_knapsack(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap,alpha,beta)
        curr_min_cap.append(min_cap)
        for ii in range(0, len(nash)):
            if math.isnan(nash[ii]):
                nash[ii]=1.0
        curr_nash.append(np.prod(nash))

        curr_satisfied.append(satisfied*1.0/len(data)*100)
        curr_balances.append(balance)
        curr_capacities.append(capacity)
        curr_n_clusters.append(n_clusters)
        curr_count_instances.append(count_instances)
        curr_tracking.append(tracking)
        if verbose:
            print("Time taken for min of capacity %d : %.3f seconds." % (min_cap, time.time() - start_time))
    return curr_min_cap,curr_nash, curr_satisfied, curr_balances, curr_capacities, curr_n_clusters, curr_count_instances, curr_tracking


def experiment_mfc_knapsack(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap, theta,alpha, beta):
    name = []
    for i in range(n_wishes):
        name.append('Pri' + str(i + 1))
    wish_df = pd.DataFrame(columns=name)

    for i in range(n_wishes):
        column_name = 'Pri' + str(i + 1)
        wish_df[column_name] = data[column_name]
    name = []
    for i in range(n_topic):
        name.append('T' + str(i + 1))
    topic_df = pd.DataFrame(columns=name)
    for i in range(n_topic):
        column_name = 'T' + str(i + 1)
        topic_df[column_name] = data[column_name]

    n_students = len(wish_df)
    wish_values = wish_df.values
    v = np.zeros((len(wish_values), n_topic))
    for i in range(n_students):
        for j in range(n_wishes):
            v[i][wish_values[i][j] - 1] = n_wishes / (j + 1)
    # Normalize
    v_max = np.max(v,axis =1)
    for i in range(n_students):
        for j in range(n_topic):
            v[i][j] = v[i][j] / v_max[i]

    w = topic_df.values * 1.0
    w_max = np.max(w, axis=0)
    # Normalize
    for i in range(n_students):
        for j in range(n_topic):
            w[i][j] = (w[i][j] * 1.0) / (w_max[j] * 1.0)
    # cost = f(v,w)
    cost = f_cost(v, w,alpha,beta)

    # Model
    Knapsack = MFC_knapsack()
    Knapsack.fit(data, wish_df.values, cost, min_cap, min_cap + diff_cap, theta, protected_att, male)
    count_clusters = 0
    for i in range(n_topic):
        if i in Knapsack.clusters.keys():
            count_clusters = count_clusters + len(Knapsack.clusters[i])



    balance, capacity, nash, satisfied, tracking = fairness_calculation(data, wish_df.values, cost, Knapsack.clusters, protected_att,
                                                              male, n_wishes)

    return count_clusters, len(Knapsack.clusters), balance, capacity, nash, satisfied, tracking

def run_experiment_mfc_knapsack(data, n_wishes, n_topic, max_of_min_cap, diff_cap = 1, protected_att = 'sex',male = 'M',theta = 0.5, alpha = 1, beta = 1,  verbose = True):
    curr_min_cap = []
    curr_nash = []
    curr_satisfied = []
    curr_balances = []
    curr_capacities = []
    curr_n_clusters = []
    curr_count_instances = []
    curr_tracking = []
    #diff_cap = 2
    for min_cap in range(2, max_of_min_cap+1, 1):
        start_time = time.time()
        count_instances, n_clusters, balance, capacity, nash, satisfied, tracking = experiment_mfc_knapsack(data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap,theta,alpha, beta)
        curr_min_cap.append(min_cap)
        for ii in range(0, len(nash)):
            if math.isnan(nash[ii]):
                nash[ii]=1.0
        curr_nash.append(np.prod(nash))

        curr_satisfied.append(satisfied*1.0/len(data)*100)
        curr_balances.append(balance)
        curr_capacities.append(capacity)
        curr_n_clusters.append(n_clusters)
        curr_count_instances.append(count_instances)
        curr_tracking.append(tracking)
        if verbose:
            print("Time taken for min of capacity %d : %.3f seconds." % (min_cap, time.time() - start_time))
    return curr_min_cap,curr_nash, curr_satisfied, curr_balances, curr_capacities, curr_n_clusters, curr_count_instances, curr_tracking

def experiment_CPLEX(dataname, data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap, alpha, beta):

    name = []
    for i in range(n_wishes):
        name.append('Pri' + str(i + 1))
    wish_df = pd.DataFrame(columns=name)

    for i in range(n_wishes):
        column_name = 'Pri' + str(i + 1)
        wish_df[column_name] = data[column_name]
    name = []
    for i in range(n_topic):
        name.append('T' + str(i + 1))
    topic_df = pd.DataFrame(columns=name)
    for i in range(n_topic):
        column_name = 'T' + str(i + 1)
        topic_df[column_name] = data[column_name]

    n_students = len(wish_df)
    wish_values = wish_df.values

    v = np.zeros((n_students, n_topic))
    for i in range(n_students):
        for j in range(n_wishes):
            v[i][wish_values[i][j] - 1] = n_wishes / (j + 1)

    # Normalize
    v_max = np.max(v, axis=1)
    for i in range(n_students):
        for j in range(n_topic):
            v[i][j] = v[i][j] / v_max[i]

    w = topic_df.values * 1.0

    w_max = np.max(w, axis=0)
    # Normalize
    for i in range(n_students):
        for j in range(n_topic):
            w[i][j] = (w[i][j] * 1.0) / (w_max[j] * 1.0)

    # cost = f(v,w)
    cost = f_cost(v, w, alpha, beta)
    #Read solution
    capacity = min_cap
    filename = 'solutions/' + dataname + '-' + str(min_cap) + '.csv'
    solution_df = pd.read_csv(filename,header=None)
    clusters = {}
    for id in range(len(solution_df.iloc[0])):
        if solution_df.iloc[0][id]>0:
            clusters_key = id
            clusters[clusters_key] = []
    for clusters_key in clusters.keys():
        for id in range(1,len(solution_df)):
            if solution_df.iloc[id][clusters_key]>0:
                clusters[clusters_key].append(id-1)

    count_clusters = 0
    for i in range(n_topic):
        if i in clusters.keys():
            count_clusters = count_clusters + len(clusters[i])



    balance, capacity, nash, satisfied, tracking = fairness_calculation(data, wish_df.values, cost, clusters, protected_att,
                                                              male, n_wishes)

    return count_clusters, len(clusters), balance, capacity, nash, satisfied, tracking


def run_experiment_CPLEX(dataname, data, n_wishes, n_topic, min_of_min_cap, max_of_min_cap, diff_cap=1, protected_att='sex', male='M',
                                 alpha=1, beta=1, verbose=True):
    curr_min_cap = []
    curr_nash = []
    curr_satisfied = []
    curr_balances = []
    curr_capacities = []
    curr_n_clusters = []
    curr_count_instances = []
    curr_tracking = []
    #diff_cap = 2
    for min_cap in range(min_of_min_cap, max_of_min_cap+1, 1):
        start_time = time.time()
        count_instances, n_clusters, balance, capacity, nash, satisfied, tracking = experiment_CPLEX(dataname, data, protected_att, male, n_wishes, n_topic, min_cap, diff_cap, alpha, beta)
        curr_min_cap.append(min_cap)
        for ii in range(0, len(nash)):
            if math.isnan(nash[ii]):
                nash[ii]=1.0
        curr_nash.append(np.prod(nash))

        curr_satisfied.append(satisfied*1.0/len(data)*100)
        curr_balances.append(balance)
        curr_capacities.append(capacity)
        curr_n_clusters.append(n_clusters)
        curr_count_instances.append(count_instances)
        curr_tracking.append(tracking)
        if verbose:
            print("Time taken for min of capacity %d : %.3f seconds." % (min_cap, time.time() - start_time))
    return curr_min_cap,curr_nash, curr_satisfied, curr_balances, curr_capacities, curr_n_clusters, curr_count_instances, curr_tracking

#Not suitable
def experiment_fair_kmedoids_knapsack(data_wish, data_org, n_clusters, min_cap, diff_cap, mcf_fairlets, mcf_fairlet_centers,
                                      weight_mcf_fairlets,max_iter,decay_lambda, n_wishes, n_topic, protected_att = 'sex',male = 'M',theta = 0.5, alpha = 1, beta = 1,  verbose = True):
    name = []
    for i in range(n_wishes):
        name.append('Pri' + str(i + 1))
    wish_df = pd.DataFrame(columns=name)

    for i in range(n_wishes):
        column_name = 'Pri' + str(i + 1)
        wish_df[column_name] = data_wish[column_name]
    name = []
    for i in range(n_topic):
        name.append('T' + str(i + 1))
    topic_df = pd.DataFrame(columns=name)
    for i in range(n_topic):
        column_name = 'T' + str(i + 1)
        topic_df[column_name] = data_wish[column_name]

    n_students = len(wish_df)
    wish_values = wish_df.values
    v = np.zeros((len(wish_values), n_topic))
    for i in range(n_students):
        for j in range(n_wishes):
            v[i][wish_values[i][j] - 1] = n_wishes / (j + 1)
    # Normalize
    v_max = np.max(v,axis =1)
    for i in range(n_students):
        for j in range(n_topic):
            v[i][j] = v[i][j] / v_max[i]

    w = topic_df.values * 1.0
    w_max = np.max(w, axis=0)
    # Normalize
    for i in range(n_students):
        for j in range(n_topic):
            w[i][j] = (w[i][j] * 1.0) / (w_max[j] * 1.0)
    # cost = f(v,w)
    cost = f_cost(v, w,alpha,beta)

    # Model
    kmedoids_capacitated = KMedoids_Knapsack(k=n_clusters, max_iter=max_iter, decay_lambda=decay_lambda)
    kmedoids_capacitated.fit([data_org[i] for i in mcf_fairlet_centers], min_cap + diff_cap, weight_mcf_fairlets)



    balance, capacity, nash, satisfied, tracking = fairness_calculation(data, wish_df.values, cost, Knapsack.clusters, protected_att,
                                                              male, n_wishes)

    return count_clusters, len(Knapsack.clusters), balance, capacity, nash, satisfied, tracking

def run_experiment_fair_kmedoids_knapsack(data_wish, data_org, n_instances, mcf_fairlets, mcf_fairlet_centers, weight_mcf_fairlets,max_iter,decay_lambda, n_wishes, n_topic, max_of_min_cap, diff_cap, protected_att = 'sex',male = 'M',theta = 0.5, alpha = 1, beta = 1,  verbose = True):
    curr_min_cap = []
    curr_nash = []
    curr_satisfied = []
    curr_balances = []
    curr_capacities = []
    curr_n_clusters = []
    curr_count_instances = []
    curr_tracking = []
    #diff_cap = 1
    for min_cap in range(2, max_of_min_cap+1, 1):
        start_time = time.time()
        n_clusters = int(math.ceil(n_instances/(min_cap+diff_cap)))
        count_instances, balance, capacity, nash, satisfied, tracking = experiment_fair_kmedoids_knapsack(data_wish, data_org, n_clusters, min_cap, diff_cap, mcf_fairlets, mcf_fairlet_centers, weight_mcf_fairlets,max_iter,decay_lambda, n_wishes, n_topic, protected_att = 'sex',male = 'M',theta = 0.5, alpha = 1, beta = 1,  verbose = True)
        curr_min_cap.append(min_cap)
        for ii in range(0, len(nash)):
            if math.isnan(nash[ii]):
                nash[ii]=1.0
        curr_nash.append(np.prod(nash))

        curr_satisfied.append(satisfied*1.0/len(data)*100)
        curr_balances.append(balance)
        curr_capacities.append(capacity)
        curr_n_clusters.append(n_clusters)
        curr_count_instances.append(count_instances)
        curr_tracking.append(tracking)
        if verbose:
            print("Time taken for min of capacity %d : %.3f seconds." % (min_cap, time.time() - start_time))
    return curr_max_cap,curr_nash, curr_satisfied, curr_balances, curr_capacities, curr_n_clusters, curr_count_instances, curr_tracking