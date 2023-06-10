import numpy as np
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import random
import math

def distance(a, b, order=2):
	"""
	Calculates the specified norm between two vectors.
	
	Args:
		a (list) : First vector
		b (list) : Second vector
		order (int) : Order of the norm to be calculated as distance
	
	Returns:
		Resultant norm value
	"""
	assert len(a) == len(b), "Length of the vectors for distance don't match."
	return np.linalg.norm(x=np.array(a)-np.array(b), ord=order)

def balance_score(data, cluster,protected_att,male):
	n_student_m = 0
	for i in cluster:
		if (data[protected_att][i] == male):
			n_student_m = n_student_m + 1
	n_student_f = len(cluster) - n_student_m
	return (min(n_student_f,n_student_m)/max(n_student_f,n_student_m))

def fairness_calculation(data, wishes,costs,clusters,protect_att,male,n_wishes):
	"""
	Checks fairness for each of the clusters
	Returns balance using the total and class counts.
	
	Args:
		data (list)
		wishes (list)
		costs (list)
		clusters (dictionary)
		protect_att (variable)
		male (variable)
		n_wishes (integer)
		
	Returns:
		balance score, capacity, Nash value, the number of satisfied students
	"""
	#for each cluster
	curr_b = []
	capacity = []
	satisfied = 0
	nash = []
	rows, cols = len(data), 3
	tracking = [([0] * cols) for i in range(rows)]
	flag = [0] * len(data)

	#print(clusters)
	for i in clusters:
		p = 0
		cost = 0
		#print(i)
		for j in clusters[i]:
			#Count the number of male
			if data[protect_att][j]==male:
				p = p +1
			#Count the number of satisfied students
			for k in range(n_wishes):
				if wishes[j][k]-1 == i:
					tracking[j][0] = wishes[j][k]  # wish
					tracking[j][1] = k  # order
					tracking[j][2] = i + 1
					if flag[j]==0:
						satisfied = satisfied + 1
						flag[j]=1
				else:
					tracking[j][0] = wishes[j][k]  # wish
					tracking[j][2] = i + 1
			#Sum of costs
			cost = cost + costs[j][i]

		q = len(clusters[i]) - p
		#print("Actual male: ", p)
		#print("Actual female: ", q)
		if p == 0 or q == 0:
			balance = 0
		else:
			balance = min(float(p / q), float(q / p))
		#balance score
		curr_b.append(balance)
		#capacity
		capacity.append(len(clusters[i]))
		#nash value
		#if cost == 0:
		#	cost = cost + 1.0
		#if math.isnan(cost):
		#	cost = 1.0
		nash.append(cost+1)
	#Print
	#print("Nash table:",nash)
	#return min(curr_b), capacity, np.prod(nash), satisfied
	return min(curr_b), capacity, nash, satisfied,tracking
def compare(array, value):
	for i in range(len(array)):
		if value==array[i]:
			return False
	return True

def generate_data(data, n, n_wish, n_topic,file=None):
	df = data.head(n)
	random.seed()
	#Generate the wishes of students
	for i in range(n_wish):
		name = 'Pri'+str(i+1)
		df[name]=None
		for j in range(len(df)):
			if i==0:
				df[name][j]= random.randint(1,n_topic)
			else:
				while True:
					value = random.randint(1,n_topic)
					array=[]
					for k in range(i):
						check_name = 'Pri'+str(k+1)
						array.append(df[check_name][j])
					if (compare(array, value)==True):
						df[name][j] = value
						break
	#Assign the weight of chosen topic
	count = []
	for i in range(n_topic):
		count.append(0)
	for i in range(n_wish):
		check_name = 'Pri'+str(i+1)
		for j in range(len(df)):
			count[df[check_name][j]-1]+=1
	for i in range(n_topic):
		name = 'T'+str(i+1)
		df[name] = None
		for j in range(len(df)):
			#Check if the instance contain topic
			check = False
			for k in range(n_wish):
				check_name = 'Pri'+str(k+1)
				if df[check_name][j] == (i+1):
					check = True
			#Assign the weight
			if check == False:
				df[name][j]=0
			else:
				if j==0:
					df[name][j]=random.randint(1,count[i])
				else:
					while True:
						weight = random.randint(1,count[i])
						array = []
						for l in range(j):
							array.append(df[name][l])
						if compare(array, weight)==True:
							df[name][j]=weight
							break
	df.to_csv(file,index = False)
	#print(count)
	return df
def f_cost(v, w, alpha=1, beta=1):
	return alpha*v + beta*w

def plot_analysis(min_cap, nash, balances, satisfied, n_clusters,step_size,file_name):
	"""
	Plots the curves for costs and balances.

	Args:
		min_cap (list)
		nash (list)
		balances (list)
		satisfied (list)
		step_size (int)
	"""
	fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6,20))
	ax[0].plot(nash, marker='8', color='blue')
	ax[0].set_xticks(list(range(0, len(min_cap), step_size)))
	ax[0].set_xticklabels(list(range(min(min_cap), max(min_cap)+1, step_size)), fontsize=12)
	#ax[0].set_yscale('log')
	ax[0].set_xlabel('Minimum of capacity')
	ax[0].set_ylabel('Nash equilibrium ')
	ax[1].plot(balances, marker='s', color='saddlebrown')
	ax[1].set_xticks(list(range(0, len(min_cap), step_size)))
	ax[1].set_xticklabels(list(range(min(min_cap), max(min_cap)+1, step_size)), fontsize=12)
	ax[1].set_xlabel('Minimum of capacity')
	ax[1].set_ylabel('Balance score w.r.t the protected attribute ')
	ax[2].plot(satisfied, marker='p', color='orange')
	ax[2].set_xticks(list(range(0, len(min_cap), step_size)))
	ax[2].set_xticklabels(list(range(min(min_cap), max(min_cap) + 1, step_size)), fontsize=12)
	ax[2].set_xlabel('Minimum of capacity')
	ax[2].set_ylabel('Satisfaction level of students'' wishes')
	ax[3].plot(n_clusters, marker='P', color='red')
	ax[3].set_xticks(list(range(0, len(min_cap), step_size)))
	ax[3].set_xticklabels(list(range(min(min_cap), max(min_cap) + 1, step_size)), fontsize=12)
	ax[3].set_xlabel('Minimum of capacity')
	ax[3].set_ylabel('Number of groups')
	plt.savefig(file_name)
	plt.show()
