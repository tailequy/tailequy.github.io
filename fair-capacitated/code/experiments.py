import time
from kcenters import KCenters
from hclust_capacitated import Hierarchical_Clustering
from KMedoids_Knapsack import KMedoids_Knapsack
from kMedoids import kMedoids

from utils import distance, balance_calculation

def run_experiments_kcenter(degrees, data, fairlets, fairlet_centers, verbose=True):
	"""
	Run experiments for decomposition.

	Args:
		degrees (int) : Maximum degree for running K-Centers
		data (list) : Data points
		fairlets (list) : Fairlets obtained from the decomposition
		fairlet_centers (list) : Fairlet centers obtained from the decomposition
		verbose (bool) : Indicator for printing progress

	Returns:
		curr_degrees (list)
		curr_costs (list)
		curr_balances (list)
	"""
	curr_degrees = []
	curr_costs = []
	curr_balances = []
	capacities = []
	## Print
	print("Len of data: ",len(data))
	##
	for degree in range(3, min(degrees+1, len(fairlet_centers)), 1):
		#print("Degree: ")
		start_time = time.time()
		
		kcenters = KCenters(k=degree)
		## Print data to check
		#checkdata = [data[i] for i in fairlet_centers]
		#print(checkdata)
		##
		kcenters.fit([data[i] for i in fairlet_centers])
		mapping = kcenters.assign()
		#Print to check
		#print('Mapping:', mapping)
		#
		
		final_clusters = []
		for fairlet_id, final_cluster in mapping:
			for point in fairlets[fairlet_id]:
				final_clusters.append((point, fairlet_centers[final_cluster]))
				
		centers = [fairlet_centers[i] for i in kcenters.centers]
		curr_degrees.append(degree)
		#curr_costs.append(max([min([distance(data[j], i) for j in centers]) for i in data]))
		cost = []
		for j in centers:
			cluster = []
			for (x, y) in final_clusters:
				if (y == j):
					cluster.append(x)
			# print('Center: ', j, ' member:', cluster)
			cost.append(sum([distance(data[j], data[i]) for i in cluster]))

		curr_costs.append(sum(cost))
		balance, cap = balance_calculation(data, centers, final_clusters)
		curr_balances.append(balance)
		capacities.append(cap)
		
		if verbose:
			print("Time taken for Degree %d - %.3f seconds."%(degree, time.time() - start_time))

	return curr_degrees, curr_costs, curr_balances, capacities


def run_experiments_kmedoids(degrees, data, fairlets, fairlet_centers, verbose=True):
	"""
	Run experiments for decomposition.

	Args:
		degrees (int) : Maximum degree for running K-Centers
		data (list) : Data points
		fairlets (list) : Fairlets obtained from the decomposition
		fairlet_centers (list) : Fairlet centers obtained from the decomposition
		verbose (bool) : Indicator for printing progress

	Returns:
		curr_degrees (list)
		curr_costs (list)
		curr_balances (list)
	"""
	curr_degrees = []
	curr_costs = []
	curr_balances = []
	capacities = []
	## Print
	print("Len of data: ", len(data))
	##
	for degree in range(3, min(degrees + 1, len(fairlet_centers)), 1):
		# print("Degree: ")
		start_time = time.time()

		kmedoids = kMedoids(n_cluster=degree)
		## Print data to check
		# checkdata = [data[i] for i in fairlet_centers]
		# print(checkdata)
		##
		kmedoids.fit([data[i] for i in fairlet_centers])
		mapping = kmedoids.assign()
		# Print to check
		# print('Mapping:', mapping)
		#

		final_clusters = []
		for fairlet_id, final_cluster in mapping:
			for point in fairlets[fairlet_id]:
				final_clusters.append((point, fairlet_centers[final_cluster]))

		centers = [fairlet_centers[i] for i in kmedoids.medoids]
		curr_degrees.append(degree)
		cost=[]
		for j in centers:
			cluster = []
			for (x,y) in final_clusters:
				if (y==j):
					cluster.append(x)
			#print('Center: ', j, ' member:', cluster)
			cost.append(sum([distance(data[j], data[i]) for i in cluster]))

		curr_costs.append(sum(cost))
		#curr_costs.append(sum([sum([distance(data[j], i) for j in centers]) for i in data]))
		balance, cap = balance_calculation(data, centers, final_clusters)
		curr_balances.append(balance)
		capacities.append(cap)

		if verbose:
			print("Time taken for Degree %d - %.3f seconds." % (degree, time.time() - start_time))

	return curr_degrees, curr_costs, curr_balances, capacities

def run_experiments_hierarchical(degrees, data, fairlets, fairlet_centers, weight_fairlets, verbose=True):
	"""
	Run experiments for decomposition.

	Args:
		degrees (int) : Maximum degree for running K-Centers
		data (list) : Data points
		fairlets (list) : Fairlets obtained from the decomposition
		fairlet_centers (list) : Fairlet centers obtained from the decomposition
		weight_fairlets (list): Fairlets' len from the decomposition
		verbose (bool) : Indicator for printing progress

	Returns:
		curr_degrees (list)
		curr_costs (list)
		curr_balances (list)
	"""
	curr_degrees = []
	curr_costs = []
	curr_balances = []
	capacities = []
	ideal_capacity = []

	for degree in range(3, min(degrees + 1, len(fairlet_centers)), 1):
		print("Degree: ",degree)
		start_time = time.time()
		# Calculate number of fairlets for each cluster
		capacity = int(len(data)*1.2/degree)
		ideal_capacity.append(capacity)

		print('Capacity:', capacity)

		hclust = Hierarchical_Clustering(k=degree)
		hclust.fit([data[i] for i in fairlet_centers], capacity, weight_fairlets,'centroid')
		mapping = hclust.assign()
#Print
		#print('mapping',mapping)
#

		final_clusters = []
		for fairlet_id, final_cluster in mapping:
			for point in fairlets[fairlet_id]:
				final_clusters.append((point, fairlet_centers[final_cluster]))

		centers = [fairlet_centers[i] for i in hclust.centers]
#Print
		#print('center:',centers)
#

		curr_degrees.append(degree)
		cost = []
		for j in centers:
			cluster = []
			for (x, y) in final_clusters:
				if (y == j):
					cluster.append(x)
			# print('Center: ', j, ' member:', cluster)
			cost.append(sum([distance(data[j], data[i]) for i in cluster]))

		curr_costs.append(sum(cost))
		#curr_costs.append(max([min([distance(data[j], i) for j in centers]) for i in data]))
		balance, cap = balance_calculation(data, centers, final_clusters)
		curr_balances.append(balance)
		capacities.append(cap)

		if verbose:
			print("Time taken for Degree %d - %.3f seconds." % (degree, time.time() - start_time))

	return curr_degrees, curr_costs, curr_balances,capacities,ideal_capacity

def run_experiments_kmedoids_knapsack(degrees, data, fairlets, fairlet_centers, weight_fairlets, max_iter, decay_lambda, verbose=True):
	"""
	Run experiments for decomposition.

	Args:
		degrees (int) : Maximum degree for running K-Centers
		data (list) : Data points
		fairlets (list) : Fairlets obtained from the decomposition
		fairlet_centers (list) : Fairlet centers obtained from the decomposition
		weight_fairlets (list): Fairlets' len from the decomposition
		verbose (bool) : Indicator for printing progress

	Returns:
		curr_degrees (list)
		curr_costs (list)
		curr_balances (list)
	"""
	curr_degrees = []
	curr_costs = []
	curr_balances = []
	capacities = []
	ideal_capacity = []

	for degree in range(3, min(degrees + 1, len(fairlet_centers)), 1):
		#print("Degree: ",degree)
		start_time = time.time()
		# Calculate number of fairlets for each cluster
		capacity = int(len(data) / degree + 0.01*len(data)/degree)

		ideal_capacity.append(capacity)

		#print('Capacity:', capacity)

		kmedoids_capacitated = KMedoids_Knapsack(k=degree,max_iter=max_iter,decay_lambda=decay_lambda)
		kmedoids_capacitated.fit([data[i] for i in fairlet_centers], capacity, weight_fairlets)
		mapping = kmedoids_capacitated.assign()
#Print
		#print('mapping',mapping)
#

		final_clusters = []
		for fairlet_id, final_cluster in mapping:
			for point in fairlets[fairlet_id]:
				final_clusters.append((point, fairlet_centers[final_cluster]))

		centers = [fairlet_centers[i] for i in kmedoids_capacitated.medoids]
#Print
		#print('center',centers)
#

		curr_degrees.append(degree)
		cost = []
		for j in centers:
			cluster = []
			for (x, y) in final_clusters:
				if (y == j):
					cluster.append(x)
			# print('Center: ', j, ' member:', cluster)
			cost.append(sum([distance(data[j], data[i]) for i in cluster]))

		curr_costs.append(sum(cost))
		#curr_costs.append(max([min([distance(data[j], i) for j in centers]) for i in data]))
		balance, cap = balance_calculation(data, centers, final_clusters)
		curr_balances.append(balance)
		capacities.append(cap)

		if verbose:
			print("Time taken for Degree %d - %.3f seconds." % (degree, time.time() - start_time))

	return curr_degrees, curr_costs, curr_balances,capacities,ideal_capacity