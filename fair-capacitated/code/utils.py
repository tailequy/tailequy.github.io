import numpy as np
import matplotlib.pyplot as plt
import heapq
#from gurobipy import *
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

def balance_calculation(data, centers, mapping):
	"""
	Checks fairness for each of the clusters defined by k-centers.
	Returns balance using the total and class counts.
	
	Args:
		data (list)
		centers (list)
		mapping (list) : tuples of the form (data, center)
		
	Returns:
		fair (dict) : key=center, value=(sum of 1's corresponding to fairness variable, number of points in center)
	"""
	fair = dict([(i, [0, 0]) for i in centers])
	for i in mapping:
		fair[i[1]][1] += 1
		if data[i[0]][0] == 1: # MARITAL
			fair[i[1]][0] += 1

	curr_b = []
	capacity = []
	for i in list(fair.keys()):
		p = fair[i][0]
		q = fair[i][1] - fair[i][0]
		if p == 0 or q == 0:
			balance = 0
		else:
			balance = min(float(p/q), float(q/p))
		curr_b.append(balance)

		### Print data
		#print(i, ",", p, ",", q,",",balance)
		capacity.append(p+q)

	return min(curr_b), capacity

def plot_analysis(degrees, costs, balances, step_size):
	"""
	Plots the curves for costs and balances.

	Args:
		degrees (list)
		costs (list)
		balances (list)
		step_size (int)
	"""
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
	ax[0].plot(costs, marker='.', color='blue')
	ax[0].set_xticks(list(range(0, len(degrees), step_size))) 
	ax[0].set_xticklabels(list(range(min(degrees), max(degrees)+1, step_size)), fontsize=12)
	ax[1].plot(balances, marker='x', color='saddlebrown')
	ax[1].set_xticks(list(range(0, len(degrees), step_size))) 
	ax[1].set_xticklabels(list(range(min(degrees), max(degrees)+1, step_size)), fontsize=12)
	plt.show()

def heap_sort(items):
	heapq.heapify(items)
	items[:] = [heapq.heappop(items) for i in range(len(items))]
	return items


def run(r,n,L,matrix,k):
	# global total_runtime, k, runtime, num_centers, m, capacity, input_file, L
	prunedMatrix = []
	centers = []
	nodes_arr = []
	for i in range(0, n):
		list = []
		for j in range(0, n):
			list.append(float(0))
		prunedMatrix.append(list)
	for i in range(0, n):
		for j in range(0, n):
			if matrix[i][j] <= r:
				prunedMatrix[i][j] = 1
			if i == j:
				prunedMatrix[i][j] = 0

	try:
		#global m, num_centers, runtime, capacity
		m = Model("mip1")

		m.setParam("MIPGap", 0.0);

		m.params.BestObjStop = k

		y = []
		for i in range(n):
			y.append(0)

		for i in range(n):
			y[i] = m.addVar(vtype=GRB.BINARY, name="y%s" % str(i + 1))

		m.setObjective(quicksum(y), GRB.MINIMIZE)

		temp_list = np.array(prunedMatrix).T.tolist()

		x = []

		for i in range(n):
			temp = []
			for j in range(n):
				temp.append(0)
			x.append(temp)

		for i in range(n):
			for j in range(n):
				x[i][j] = m.addVar(vtype=GRB.BINARY, name="x%s%s" % (str(i + 1), str(j + 1)))

		temp_list_2 = np.array(x).T.tolist()

		if L == 'NA':
			for i in range(n):
				m.addConstr(quicksum(temp_list_2[i]) <= capacity[i])
		else:
			L = int(L)
			for i in range(n):
				m.addConstr(quicksum(temp_list_2[i]) <= L)

		for i in range(n):
			for j in range(n):
				m.addConstr(x[i][j] <= y[j])

		for i in range(n):
			for j in range(n):
				m.addConstr(x[i][j] <= prunedMatrix[i][j])

		for i in range(n):
			m.addConstr(quicksum(x[i]) == 1 - y[i])

		m.optimize()
		runtime = m.Runtime
		# print("The run time is %f" % runtime)
		# print("Obj:", m.objVal)

		dom_set_size = 0
		solution = []
		assignment = []
		center = 0
		vertex_j = 1
		vertex_i = 1
		for v in m.getVars():
			varName = v.varName
			if varName[0] == 'y':
				if v.x >= 0.9:
					dom_set_size = dom_set_size + 1
					solution.append(varName[1:])
			else:
				if vertex_j <= n:
					# if v.x == 1.0:
					if v.x >= 0.9:
						assignment.append([vertex_i, vertex_j])
				else:
					vertex_i = vertex_i + 1
					vertex_j = 1
				vertex_j = vertex_j + 1
		# print("Cap. dom. set cardinality: " + str(dom_set_size))
		solution = [int(i) for i in solution]
		# print("solution: " + str(solution))
		# print("assignment: " + str(assignment))

		# print('{"instance": "%s",' % input_file)
		# print('"centers": [')
		counter = 0
		for center in solution:
			counter = counter + 1
			nodes = []
			for node in assignment:
				if node[1] == center:
					nodes.append(node[0])
			# if counter == len(solution):
			# print('{ "center": ' + str(center) + ', "nodes": ' + str(nodes) + '}')
			# else:

			# print('{ "center": ' + str(center) + ', "nodes": ' + str(nodes) + '},')
			# print(']}')
			#print(center)
			#print(nodes)
			centers.append(center)
			nodes_arr.append(nodes)

		num_centers = dom_set_size
	#        num_centers = m.objVal

	except GurobiError:
		print("Error reported")
	return centers, nodes_arr, num_centers