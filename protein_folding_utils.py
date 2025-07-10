"""
Protein Folding Utilities for QAOA Implementation

HP-lattice Coordinate-based HP-lattice model
Written by Hanna Linn, hannlinn@chalmers.se
Based on Lucas Knuthsons code.

If used, please cite Irbäck et al. 2022: 
https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043013

This module contains utility functions and classes for implementing the HP lattice
model for protein folding optimization using QAOA (Quantum Approximate Optimization Algorithm).
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, NullFormatter, MaxNLocator
from matplotlib import cm
import networkx as nx
import math
from itertools import product
from datetime import datetime
from scipy.io import savemat
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
import pennylane as qml
from pennylane import numpy as np

class CoordinateBased_HPLattice:
	'''
	Class for one instance of the 2D lattice HP-model to then be fed into the quantum simulation.

	Class variables in order of creation [type]:
	After init:
		- dim_lattice = dimensions of the lattice (rows, columns) [tuple of two ints]
		- lambda_vector = penalty terms for the energy terms [tuple of three ints]
		- sequence = Given sequence of the problem. Sequence format: H is 1 and 0 is P, e.g., [H, P, P, H] = [1, 0, 0, 1]. [list of int/binaries]
		- Q = dict with energy info [dictionary]
		- bit_name_list = names of the qubits saved in a list. Name format: (node [tuple], seq [int]) [list of tuples]
		- num_bits = number of bits in the instance [int]
		- O_energies = one-body energies [list of floats]
		- T_energies = two-body energies [list of Numpy Arrays with floats]
		- Dn = Dn vector [number of nodes for placements for each amino acid in sequence] [list] DOES NOT WORK FOR RECTANGULAR LATTICES!!!

	After function call:
		- feasible_set = Numpy arrays of all feasible bitstrings solutions in one-hot encoding [list of Numpy Arrays]
		- solution_set = Numpy arrays of all bitstring solutions in one-hot encoding [list of Numpy Arrays]

	'''

	def __init__(self, sequence: List[int], dim_lattice: Tuple[int, int], 
				lambda_vector: Tuple[float, float, float], max_index: Optional[int] = None, 
				verbose: bool = False) -> None:
		"""
		Initialize the HP lattice protein folding problem.
		
		Args:
			sequence: List of 1s (hydrophobic) and 0s (polar) representing the protein
			dim_lattice: Tuple (rows, cols) defining the lattice dimensions
			lambda_vector: Tuple of penalty coefficients (lambda1, lambda2, lambda3)
			max_index: Maximum index for solution enumeration (default: None for all)
			verbose: Whether to print detailed information
		"""
		self.sequence = sequence
		self.dim_lattice = dim_lattice
		self.lambda_vector = lambda_vector
		self.max_index = max_index
		
		# Initialize derived properties
		self.bit_name_list = self.get_bit_names()
		self.num_bits = len(self.bit_name_list)
		self.Q = self.make_Q()
		self.O_energies = self.get_O_energies()
		self.T_energies = self.get_T_energies()
		self.Dn = self.get_Dn()
		
		# Calculate solution sets
		self.solution_set = self.get_solution_set()
		self.feasible_set = self.get_feasible_set()
		
		if verbose:
			self.print_summary()
	
	def __str__(self) -> str:
		"""
		String representation of the CoordinateBased_HPLattice object.
		Returns a string summarizing the one-body and two-body energies.

		Returns:
			str: String representation showing one-body (O) and two-body (T) energies
		"""
		return '\nO:\n' + str(self.O_energies) + '\nT:\n' + str(self.T_energies)

	def print_summary(self) -> None:
		"""
		Print a comprehensive summary of the protein folding problem setup.
		
		Displays key information including the protein sequence, lattice dimensions,
		problem size, and constraint statistics. Useful for problem diagnosis
		and parameter verification.
		
		Returns:
			None: Function prints to stdout
		
		Prints:
			- Protein sequence (H/P pattern)
			- Lattice dimensions (rows × columns)  
			- Total number of qubits/variables
			- Total number of possible solutions (2^n)
			- Number of physically feasible solutions
			- Percentage of feasible solutions (constraint tightness metric)
		"""
		print(f"Protein sequence: {self.sequence}")
		print(f"Lattice dimensions: {self.dim_lattice}")
		print(f"Number of qubits: {self.num_bits}")
		print(f"Total solutions: {len(self.solution_set)}")
		print(f"Feasible solutions: {len(self.feasible_set)}")
		print(f"Feasible percentage: {self.get_feasible_percentage():.2f}%")

	def get_H_indices(self):
		'''
		The following lists are lists of the indices where
		Hs are positioned.
		Used in make_Q.
		Based on code by: Lucas Knuthson
		'''
		H_index_even = [i for i in range(len(self.sequence)) if self.sequence[i] == 1 and i % 2 == 0]
		H_index_odd = [i for i in range(len(self.sequence)) if self.sequence[i] == 1 and i % 2 == 1]
		return H_index_even, H_index_odd

	def combos_of_H(self):
		'''
		Used in make_Q.
		Based on code by: Lucas Knuthson
		'''
		H_index_even, H_index_odd = self.get_H_indices()
		H_combos = []
		for even in H_index_even:
			for odd in H_index_odd:
				H_combos.append((even, odd))
		return H_combos

	def split_evenodd(self):
		'''
		Split the sequence into a lists of odd and even beads.
		Cates = categories.
		Used in make_Q.
		Based on code by: Lucas Knuthson
		'''
		cates_even = [i for i in range(len(self.sequence)) if i%2 == 0]
		cates_odd = [i for i in range(len(self.sequence)) if i%2 == 1]
		return cates_even, cates_odd

	def make_Q(self, verbose = False):
		'''
		Q is the interactions in the ising model.
		Two-body energy: (q_1, q_2) = value
		One-body energy: (q_1, q_1) = value

		bit format: (node, seq. index)
		Node format: (row, col)
		Based on code by: Lucas Knuthson
		'''
		from collections import defaultdict
		Q = defaultdict(int)

		cates_even, cates_odd = self.split_evenodd()

		G = nx.grid_2d_graph(self.dim_lattice[0], self.dim_lattice[1]) # makes a lattice as a graph

		# EPH
		H_combos = self.combos_of_H()
		count_HP = 0
		#print('HP')
		for u,v in G.edges():
			for x,y in H_combos:
				if (x-y)**2 > 4:
					if sum(u) % 2 != 1 and sum(v) % 2 == 1:
						Q[((u,x), (v,y))] -= 1
						count_HP += 1
						#print('-', ((u,x), (v,y)))
					elif sum(u) % 2 == 1 and sum(v) % 2 != 1:
						Q[((v,x), (u,y))] -= 1
						count_HP += 1
						#print('-', ((v,x), (u,y)))
					
		# Sums over the squared sums, lambda 1
		count_onper = 0
		#print('Sums over the squared sums')
		#even
		for i in cates_even:
			# One body
			for u in G.nodes():
				if sum(u) % 2 != 1:
					Q[((u,i), (u,i))] -= 1*self.lambda_vector[0]
					count_onper += 1
					#print('-', ((u,i), (u,i)))

			# Two body
			for u in G.nodes():
				for v in G.nodes():
					if u != v and (sum(u) % 2 != 1 and sum(v) % 2 != 1) :
						Q[((u,i),(v,i))] += 2*self.lambda_vector[0]
						count_onper += 1
						#print(((u,i),(v,i)))

		#odd
		for i in cates_odd:
			# One body
			for u in G.nodes():
				if sum(u) % 2 == 1:
					Q[((u,i),(u,i))] -= 1*self.lambda_vector[0]
					count_onper += 1
					#print('-', ((u,i),(u,i)))

			# Two body
			for u in G.nodes():
				for v in G.nodes():
					if u != v and (sum(u) % 2 == 1 and sum(v) % 2 == 1):
						Q[((u,i),(v,i))] += 2*self.lambda_vector[0]
						count_onper += 1
						#print(((u,i),(v,i)))

		# self-avoidance, lambda 2
		#print('self-avoidance')
		count_sa = 0
		for u in G.nodes():
			if sum(u) % 2 != 1: # even beads
				for x in cates_even:
					for y in cates_even:
						if x != y and x < y:
							Q[((u,x), (u,y))] += 1*self.lambda_vector[1]
							count_sa += 1
							#print(((u,x), (u,y)))
			elif sum(u) % 2 == 1: # odd beads
				for x in cates_odd:
					for y in cates_odd:
						if x != y and x < y:
							Q[((u,x), (u,y))] += 1*self.lambda_vector[1]
							count_sa += 1
							#print(((u,x), (u,y)))

		# Connectivity sums, lambda 3
		# Even
		#print('Connectivity sums')
		count_con = 0
		for i in cates_even:
			for u in G.nodes():
				for v in G.nodes():
					if (((u,v) in G.edges()) == False and ((v,u) in G.edges()) == False) and u != v:
						if sum(u) % 2 != 1 and sum(v) % 2 == 1 and len(self.sequence) % 2 == 0:
							count_con += 1
							#print(((u,i), (v,i+1)))
							Q[((u,i), (v,i+1))] += 1*self.lambda_vector[2]

						elif sum(u) % 2 != 1 and sum(v) % 2 == 1 and len(self.sequence) % 2 == 1:
							if i != cates_even[-1]:
								Q[((u,i), (v,i+1))] += 1*self.lambda_vector[2]
								count_con += 1
								#print(((u,i), (v,i+1)))
		# Odd
		for i in cates_odd:
			for u in G.nodes():
				for v in G.nodes():
					if (((u,v) in G.edges()) == False and ((v,u) in G.edges()) == False) and u != v:
						if (sum(u) % 2 != 1 and sum(v) % 2 == 1) and len(self.sequence) % 2 == 1:
							Q[((u,i+1), (v,i))] += 1*self.lambda_vector[2]
							count_con += 1
							#print(((u,i+1), (v,i)))

						elif (sum(u) % 2 != 1 and sum(v) % 2 == 1) and len(self.sequence) % 2 == 0:
							if i != cates_odd[-1]:
								count_con += 1
								#print(((u,i+1), (v,i)))
								Q[((u,i+1), (v,i))] += 1*self.lambda_vector[2]

		if verbose:
			print('Counts:')
			print('HP: ', count_HP)
			print('onper, lambda 1: ', count_onper)
			print('self-avoidance, lambda 2: ', count_sa)
			print('connectivity, lambda 3: ', count_con)

		Q = dict(Q) # not a defaultdict anymore to not be able to grow by error
		return Q

	def get_node_list(self, verbose = False):
		'''
		Returns a list of the nodes in the right order: snakey!
		Verbose will print resulting list and saves a .png of the graph.
		'''
		node_list = []
		(Lrow, Lcol) = self.dim_lattice
		G = nx.grid_2d_graph(Lrow, Lcol)

		for row in range(Lrow):
			start_index = row * Lcol
			if row % 2 == 0: # even row is forward
				node_list.extend(list(G.nodes())[start_index:start_index + Lcol])
			if row % 2 == 1: # odd row is backward
				node_list.extend(reversed(list(G.nodes())[start_index:start_index + Lcol]))
		if verbose:
			nx.draw(G, with_labels=True)
			plt.savefig('Lattice')
			print(node_list)
		return node_list

	def get_bit_names(self):
		'''
		Returns a list of all the bitnames in the form (node (row, col), seq) in the right order.
		'''

		seq_index = range(len(self.sequence))
		node_list = self.get_node_list(verbose = False)
		bit_name_list = []
		L_2 = int(self.dim_lattice[0]*self.dim_lattice[1])
		nodes_even = [x for x in range(L_2) if x % 2 == 0]
		nodes_odd = [x for x in range(L_2) if x % 2 != 0]
		# for all even nodes with first index aso
		for f in seq_index:
			if f % 2 == 0:
				for s in nodes_even:
					bit_name_list.append((node_list[s], f))
			if f % 2 == 1:
				for s in nodes_odd:
					bit_name_list.append((node_list[s], f))
		return bit_name_list

	def get_O_energies(self):
		'''
		Get the one-body energies for the Hamiltonian.
		'''
		O_energies = []
		for bit in self.bit_name_list:
			try:
				O_energies.append(self.Q[(bit, bit)])
			except:
				pass
		return O_energies

	def get_T_energies(self):
		'''
		Get the two-body energies for the Hamiltonian.
		'''
		T_energies = np.zeros((self.num_bits, self.num_bits))

		for j in range(self.num_bits):
			for k in range(self.num_bits):
				if j == k:
					T_energies[j,k] = 0
				else:
					try:
						T_energies[j,k] = self.Q[self.bit_name_list[j], self.bit_name_list[k]]
						if j > k:
							T_energies[k,j] = self.Q[self.bit_name_list[j], self.bit_name_list[k]]
					except:
						pass

		T_energies = np.triu(T_energies) # delete lower triangle
		T_energies = T_energies + T_energies.T - np.diag(np.diag(T_energies)) # copy upper triangle to lower triangle
		return T_energies

	def get_Dn(self):
		'''
		Cardinality vector. Used for XY-mixer.
		'''
		D = []
		for seq in range(len(self.sequence)):
			if seq % 2 == 0:
				D.append(math.ceil((self.dim_lattice[0]*self.dim_lattice[1])/2))
			if seq % 2 == 1:
				D.append(math.floor((self.dim_lattice[0]*self.dim_lattice[1])/2))
		return D

	def get_feasible_percentage(self):
		return 100*(len(self.feasible_set)/len(self.solution_set))

	def get_solution_set(self):
		'''
		Input: Number of bits.
		Output: Numpy arrays of dimensions (1, num_bits) in a list of all possible bitstrings.
		'''
		return [np.array(i) for i in product([0, 1], repeat = self.num_bits)]

	def get_feasible_set(self):
		'''
		Output: Numpy arrays of all feasible solutions, in a list.
		Hamming distance is 1 at each position.
		'''
		feasible_list = []
		index_list = []
		start = 0
		for rot in self.Dn:
			stop = start + rot
			index_perm = [x for x in range(start, stop)]
			index_list.append(index_perm)
			start = stop
		comb = list(product(*index_list))
		for i in comb:
			state = np.zeros(self.num_bits)
			for j in i:
				state[j] = 1

			feasible = True

			# same node and on?
			for b in range(self.num_bits):
				
				node1 = self.bit_name_list[b][0]
				node2 = self.bit_name_list[(b + self.dim_lattice[0]*self.dim_lattice[1]) % self.num_bits][0]
				if (node1 == node2) and state[b] and state[(b + self.dim_lattice[0]*self.dim_lattice[1]) % self.num_bits]:
					feasible = False
					break
			
			# longer distance than 1 manhattan distance
			if feasible:
				for bit1 in range(len(state)):
					found = False
					if state[bit1] == 1:
						for bit2 in range(bit1+1, len(state)):
							if state[bit2] == 1 and not found:
								found = True
								node1 = self.bit_name_list[bit1][0]
								node2 = self.bit_name_list[bit2][0]
								if self.manhattan_dist(node1, node2) > 1:
									feasible = False
									break
						else:
							continue
			if feasible:
				feasible_list.append(state)
		return feasible_list

	def manhattan_dist(self, node1, node2):
		distance = 0
		for node1_i, node2_i in zip(node1, node2):
			distance += abs(node1_i - node2_i)
		return int(distance)

	def calc_solution_sets(self):
		'''
		May take a while.
		'''
		self.feasible_set = self.get_feasible_set()
		self.solution_set = self.get_solution_set()

	def bit2energy(self, bit_array):
		Oe = np.dot(bit_array, self.O_energies)

		Te = 0
		for j,bit in enumerate(self.bit_name_list):
			for k,bit in enumerate(self.bit_name_list):
				if bit_array[j] == 1.0 and bit_array[k] == 1.0:
					Te += self.T_energies[j,k]
					
		energy = Oe + Te
		return energy

	def energy_of_set(self, feasible = False, verbose = False):
		energy_list = []
		labels = []
		mem = 1000000
		lowest_energy_bitstring = None
		if feasible:
			set_ = self.feasible_set
		else:
			set_ = self.solution_set
		for i in range(len(set_)):
			energy = self.bit2energy(set_[i])

			if verbose and (i%1000==0):
				print('Progress in energy calculations: ', round(100*i/len(set_), 1), '%%')
			try:
				energy = self.bit2energy(set_[i])
				if energy < mem:
					lowest_energy_bitstring = [set_[i], i, energy]
					mem = energy
				label = str(set_[i])
				label = label.replace(',', '')
				label = label.replace(' ', '')
				label = label.replace('.', '')
				labels.append(label)
			except:
				energy = 1000000
				if not feasible:
					label = str(set_[i])
					label = label.replace(',', '')
					label = label.replace(' ', '')
					label = label.replace('.', '')
					labels.append(label)
			energy_list.append(energy)
		print('Done!')
		return energy_list, labels, lowest_energy_bitstring

	def viz_solution_set(self, energy_for_set, labels, lowest_energy, title = '', sort = False):
		x = np.array(energy_for_set)

		if sort:
			sort_idx = x.argsort()
			x = x[sort_idx]
			labels = np.array(labels)[sort_idx]

		fig, ax = plt.subplots(figsize=(18, 4))
		plt.style.use("seaborn")
		matplotlib.rc('xtick', labelsize=12)
		ax.bar(range(len(energy_for_set)), x, tick_label = labels)

		if not sort:
			theoretical_lowest_idx = lowest_energy[1]
			ax.get_xticklabels()[theoretical_lowest_idx].set_color("green")
		else:
			ax.get_xticklabels()[0].set_color("green")

		plt.xlabel('Bitstrings')
		plt.xticks(rotation=85)
		plt.ylabel('Classic energy')
		plt.title(r'Classic energy for ' + title + ' bitstrings')

	def bit2coord(self, bit):
		x = []
		y = []
		for i in range(len(bit)):
			if int(bit[i]):
				x.append(self.bit_name_list[i][0][0])
				y.append(self.bit_name_list[i][0][1])
		return x, y

	def viz_lattice(self, bit):
		x_grid = range(self.dim_lattice[0])
		x_grid = [-x for x in x_grid]
		y_grid = range(self.dim_lattice[0])
		protein_grid = [0] * self.dim_lattice[0]

		plt.scatter(y_grid, x_grid, c=protein_grid, cmap='Greys', s=10)

		x, y = self.bit2coord(bit)
		#x = [0, 0, 1, 1]
		x = [-x for x in x]
		#y = [0, 1, 1, 0]
		
		plt.plot(y, x, 'k-', zorder=0)  # straight lines
		# large dots, set zorder=3 to draw the dots on top of the lines
		plt.scatter(y, x, c=self.sequence, cmap='coolwarm', s=1500, zorder=3) 

		plt.margins(0.2) # enough margin so that the large scatter dots don't touch the borders
		plt.gca().set_aspect('equal') # equal distances in x and y direction

		plt.axis('on')
		ax = plt.gca()
		ax.xaxis.set_major_locator(MultipleLocator(1))
		ax.xaxis.set_major_formatter(NullFormatter())
		ax.yaxis.set_major_locator(MultipleLocator(1))
		ax.yaxis.set_major_formatter(NullFormatter())
		ax.tick_params(axis='both', length=0)
		plt.grid(True, ls=':')
		plt.title(str(bit))

		for i in range(len(self.sequence)):
			plt.annotate(i, (y[i], x[i]), color='white', fontsize=24, weight='bold', ha='center')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sort_over_threshold(x: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Sort and filter values above a threshold.
	
	This function filters an array to keep only values above a given threshold,
	then returns both the filtered values and their original indices in sorted order.
	
	Args:
		x (np.ndarray): Array of values to filter and sort
		threshold (float): Minimum threshold value for filtering
		
	Returns:
		Tuple[np.ndarray, np.ndarray]: A tuple containing:
			- filtered_values: Values above threshold sorted in ascending order
			- indices: Original indices of the filtered values in the input array
			
	Example:
		>>> x = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
		>>> values, indices = sort_over_threshold(x, 0.25)
		>>> print(values)  # [0.3, 0.5, 0.8]
		>>> print(indices)  # [4, 1, 3]
	"""
	mask = x > threshold
	sort_idx = x.argsort()
	mask_idx = sort_idx[mask[sort_idx]]
	return x[mask_idx], mask_idx

def plot_probs_with_energy(probs: np.ndarray, num_qubits: int, H_cost: Any, 
						   ground_states_i: np.ndarray, new_fig: bool = True, 
						   save: bool = False, name: str = '', threshold: float = 0.001) -> Optional[plt.Figure]:
	"""
	Plot probabilities with corresponding energies and bit strings.
	
	This function creates a bar chart showing the probability of measuring
	different quantum states, colored by their classical energy. Ground states
	are highlighted in green. States below the probability threshold are filtered out.
	
	Args:
		probs (np.ndarray): Array of measurement probabilities for each quantum state
		num_qubits (int): Number of qubits in the quantum system
		H_cost (Any): Cost Hamiltonian for energy calculations (PennyLane operator)
		ground_states_i (np.ndarray): Indices of ground states
		new_fig (bool, optional): Whether to create a new figure. Default: True
		save (bool, optional): Whether to save the plot to file. Default: False
		name (str, optional): Name prefix for saved file. Default: ''
		threshold (float, optional): Minimum probability threshold for display. Default: 0.001
		
	Returns:
		Optional[plt.Figure]: The figure object containing the plot, or None if no data to plot
		
	Note:
		- States with probability below threshold are filtered out to avoid clutter
		- Ground states are marked with green x-axis labels
		- If no states meet the threshold, function prints message and returns early
		- Energy values are used to color-code the probability bars
		
	Example:
		>>> fig = plot_probs_with_energy(probabilities, 4, hamiltonian, ground_indices)
		>>> plt.show()  # Display the plot
	"""
	x, mask_idx = sort_over_threshold(probs, threshold)
	
	if len(x) < 1:
		print(f'No solutions with probability over threshold: {threshold}')
		# Create an empty figure to avoid breaking the notebook
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.text(0.5, 0.5, f'No solutions with probability > {threshold}', 
				ha='center', va='center', transform=ax.transAxes, fontsize=16)
		ax.set_title('Probability Distribution - No Solutions Above Threshold')
		plt.show()
		return fig
	
	indices = np.arange(len(probs), dtype=int)[mask_idx]
	labels_y = index_set2bit_string(indices, num_qubits)
	y = np.array([x for x in energies_of_set(indices, H_cost, num_qubits)])
	
	# Additional check in case y is empty
	if len(y) < 1:
		print('No energy values to plot')
		return None
	
	# Create figure (avoid constrained_layout which can cause warnings)
	fig, ax = plt.subplots(figsize=(15, 7))
	try:
		plt.tight_layout(pad=2.0)  # Use tight_layout instead
	except Exception:
		pass  # Continue without layout adjustment if it fails
	
	# Handle case where all energies are the same (avoid colormap issues)
	if len(set(y)) == 1:
		colors = ['skyblue'] * len(y)
		cmap = None
		norm = None
	else:
		cmap = plt.get_cmap('coolwarm', len(y))
		norm = matplotlib.colors.Normalize(vmin=np.min(y), vmax=np.max(y))
		norms = norm(y)
		colors = matplotlib.cm.coolwarm(norms)
	
	bars = ax.bar(range(len(x)), x, color=colors)
	ax.set_xticks(range(len(y)))
	ax.set_xticklabels(labels_y)
	
	# Create colorbar only if we have energy variation
	if cmap is not None and norm is not None:
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		cbar = plt.colorbar(sm, ax=ax)
		cbar.set_label('Classical Energy', labelpad=15, fontsize=20)
	
	# Mark ground states in green
	ground_energy = energy_of_index(ground_states_i[0], H_cost)
	for i, energy in enumerate(y):
		if round(float(energy), 4) == round(ground_energy, 4):
			ax.get_xticklabels()[i].set_color("green")
	
	ax.set_ylabel('Probability', fontsize=20)
	ax.set_title('Probability of measuring bit strings with energy', fontsize=25)
	ax.tick_params(axis='x', labelsize=16, rotation=85)
	ax.tick_params(axis='y', labelsize=17)
	
	if save:
		now = datetime.now()
		date_time = now.strftime("%m_%d_%Y")
		plt.savefig(f'{name}_probs_with_energy_{date_time}.pdf')

	return fig

def energies_of_set(_set: Union[List[np.ndarray], List[int]], H_cost: Any, num_qubits: int) -> np.ndarray:
	"""
	Calculate energies for a set of quantum states.
	
	This function computes the energy eigenvalues for a collection of quantum states,
	which can be provided either as bit arrays or computational basis indices.
	
	Args:
		_set (Union[List[np.ndarray], List[int]]): Collection of quantum states, either as:
			- List of bit arrays (e.g., [[0,1,0], [1,0,1]])
			- List of computational basis indices (e.g., [2, 5])
		H_cost (Any): Cost Hamiltonian operator for energy calculations
		num_qubits (int): Number of qubits in the quantum system
		
	Returns:
		np.ndarray: Array of energy eigenvalues corresponding to each state in the set
		
	Note:
		The function automatically detects the input format:
		- If input elements have length >= num_qubits, treats as bit arrays
		- Otherwise treats as computational basis indices
		
	Example:
		>>> bit_arrays = [[0,1,0], [1,0,1]]
		>>> energies = energies_of_set(bit_arrays, hamiltonian, 3)
		>>> # Returns energies for states |010⟩ and |101⟩
		
		>>> indices = [2, 5]  # Same states as above
		>>> energies = energies_of_set(indices, hamiltonian, 3)
		>>> # Returns same energies
	"""
	try:
		if len(_set[0]) >= num_qubits:
			indices = bit_array_set2indices(_set)
	except:
		indices = _set
	
	energies_index_states = get_energies_index_states(H_cost)
	energies = np.take(energies_index_states, indices)
	return energies

def get_energies_index_states(H_cost: Any) -> np.ndarray:
	"""
	Get energy eigenvalues for all computational basis states.
	
	This function computes the energy eigenvalues for every possible computational
	basis state (|00...0⟩ to |11...1⟩) by extracting the diagonal elements of
	the Hamiltonian matrix representation.
	
	Args:
		H_cost (Any): Cost Hamiltonian operator (typically a PennyLane operator)
		
	Returns:
		np.ndarray: Real-valued energy eigenvalues for all 2^n computational basis states,
					where n is the number of qubits. Values are rounded to 8 decimal 
					places for numerical stability.
					
	Note:
		- Uses the sparse matrix representation of the Hamiltonian for efficiency
		- The rounding to 8 decimal places helps avoid floating-point precision issues
		- Result array index i corresponds to computational basis state |i⟩
		
	Example:
		For a 2-qubit system, returns energies for states:
		[E(|00⟩), E(|01⟩), E(|10⟩), E(|11⟩)]
	"""
	matrix = H_cost.sparse_matrix()
	return matrix.diagonal().real.round(8)

def energy_of_index(index: int, H_cost: Any) -> float:
	"""
	Get energy eigenvalue of a specific computational basis state.
	
	This function returns the energy eigenvalue for a single computational basis
	state specified by its index in the standard binary ordering.
	
	Args:
		index (int): Index of the computational basis state (0 to 2^n - 1)
					where n is the number of qubits
		H_cost (Any): Cost Hamiltonian operator for energy calculations
		
	Returns:
		float: Energy eigenvalue of the specified computational basis state
		
	Example:
		For a 3-qubit system:
		- energy_of_index(0, H_cost) returns energy of |000⟩
		- energy_of_index(5, H_cost) returns energy of |101⟩
		- energy_of_index(7, H_cost) returns energy of |111⟩
		
	Note:
		This is a convenience function that calls get_energies_index_states()
		and extracts the specific index. For multiple lookups, it's more
		efficient to call get_energies_index_states() once and index directly.
	"""
	energies = get_energies_index_states(H_cost)
	return energies[index]

def index_set2bit_string(indices: Union[List[int], np.ndarray], num_qubits: int) -> List[str]:
	"""
	Convert computational basis state indices to bit string representations.
	
	This function converts a collection of quantum state indices into their
	corresponding binary string representations, useful for labeling and visualization.
	
	Args:
		indices (Union[List[int], np.ndarray]): Collection of computational basis state indices
		num_qubits (int): Number of qubits (determines bit string length)
		
	Returns:
		List[str]: List of bit strings with leading zeros to ensure uniform length
		
	Example:
		>>> index_set2bit_string([0, 5, 6], 3)
		['000', '101', '110']
		
		>>> index_set2bit_string([1, 4, 7], 4)  
		['0001', '0100', '0111']
		
	Note:
		- Bit strings are zero-padded to num_qubits length
		- Uses standard binary representation where leftmost bit is most significant
		- Index 0 corresponds to |00...0⟩, index 2^n-1 corresponds to |11...1⟩
	"""
	bit_strings = []
	for index in indices:
		bit_string = format(index, f'0{num_qubits}b')
		bit_strings.append(bit_string)
	return bit_strings

def bit_array_set2indices(bit_arrays: List[np.ndarray]) -> List[int]:
	"""
	Convert bit arrays to computational basis state indices.
	
	This function converts a collection of binary arrays representing quantum states
	into their corresponding integer indices in the computational basis.
	
	Args:
		bit_arrays (List[np.ndarray]): List of binary arrays representing quantum states
		
	Returns:
		List[int]: List of corresponding computational basis state indices
		
	Example:
		>>> bit_arrays = [np.array([0,0,1]), np.array([1,0,1])]
		>>> indices = bit_array_set2indices(bit_arrays)
		>>> print(indices)  # [1, 5]
		
		>>> # Explanation:
		>>> # [0,0,1] = 0*4 + 0*2 + 1*1 = 1
		>>> # [1,0,1] = 1*4 + 0*2 + 1*1 = 5
		
	Note:
		- Assumes binary arrays represent states in computational basis
		- Leftmost bit is most significant (standard binary convention)
		- Each bit array should have the same length (number of qubits)
		- Uses string conversion for robust binary-to-decimal conversion
	"""
	indices = []
	for bit_array in bit_arrays:
		index = int(''.join(map(str, bit_array.astype(int))), 2)
		indices.append(index)
	return indices

def get_ground_states_i(H_cost: Any) -> np.ndarray:
	"""
	Find all ground state indices for a given Hamiltonian.
	
	Identifies all computational basis states that achieve the minimum energy
	eigenvalue. In cases of degeneracy, multiple states may share the ground
	state energy.
	
	Args:
		H_cost (Any): Cost Hamiltonian operator (typically PennyLane operator)
		
	Returns:
		np.ndarray: Array of integer indices corresponding to ground states
					in the computational basis |0⟩, |1⟩, ..., |2^n-1⟩
					
	Example:
		For a 2-qubit system where states |01⟩ and |10⟩ both have minimum energy:
		>>> ground_indices = get_ground_states_i(hamiltonian)
		>>> print(ground_indices)  # [1, 2] (indices for |01⟩ and |10⟩)
		
	Note:
		- Returns all degenerate ground states, not just one
		- Uses numerical precision to handle floating-point comparisons
		- Essential for analyzing optimization landscapes and solution quality
	"""
	energies = get_energies_index_states(H_cost)
	min_energy = np.min(energies)
	ground_indices = np.where(energies == min_energy)[0]
	return ground_indices

def get_ground_states_energy_and_indices(feasible_set: List[np.ndarray], H_cost: Any) -> Tuple[float, np.ndarray]:
	"""
	Find ground state energy and indices from a feasible solution set.
	
	This function is specifically designed for protein folding problems where
	only a subset of quantum states represent physically valid protein configurations.
	Unlike get_ground_states_i() which considers all 2^n states, this function
	only evaluates the provided feasible configurations.
	
	Args:
		feasible_set (List[np.ndarray]): List of binary arrays representing feasible protein 
							configurations (states that satisfy physical constraints like
							no overlaps, connectivity, and lattice placement rules)
		H_cost (Any): Cost Hamiltonian operator for energy evaluation
		
	Returns:
		Tuple[float, np.ndarray]: A tuple containing:
			- min_energy (float): Lowest energy among feasible states
			- ground_indices (np.ndarray): Indices in feasible_set that achieve min_energy
			
	Example:
		>>> feasible_states = [np.array([1,0,1,0]), np.array([0,1,0,1])]
		>>> min_energy, ground_idx = get_ground_states_energy_and_indices(feasible_states, H_cost)
		>>> print(f"Ground energy: {min_energy}, found at feasible_set[{ground_idx}]")
			
	Note:
		- This is the preferred method for constrained optimization problems
		- Returns indices into feasible_set, not computational basis indices
		- Essential for protein folding where most states violate physical constraints
		- Handles degenerate ground states by returning all optimal indices
	"""
	# Calculate energies for all feasible states
	feasible_energies = []
	for state in feasible_set:
		# Convert bit array to index
		index = int(''.join(map(str, state.astype(int))), 2)
		energy = energy_of_index(index, H_cost)
		feasible_energies.append(energy)
	
	feasible_energies = np.array(feasible_energies)
	min_energy = np.min(feasible_energies)
	ground_indices = np.where(feasible_energies == min_energy)[0]
	
	return min_energy, ground_indices

def grid_search(start_gamma: float, stop_gamma: float, num_points_gamma: int,
				start_beta: float, stop_beta: float, num_points_beta: int,
				heuristic: Callable[[np.ndarray], float], plot: bool = True, 
				above: bool = False, save: bool = False, 
				matplot_save: bool = False) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Optional[plt.Figure]]:
	"""
	Perform grid search optimization for QAOA parameters (p=1 circuits).
	
	Systematically evaluates a heuristic function across a 2D grid of gamma (cost)
	and beta (mixer) parameters to find optimal QAOA settings. This is the standard
	approach for p=1 QAOA optimization where each layer has one gamma and one beta.
	
	Args:
		start_gamma (float): Lower bound for gamma (cost) parameter range
		stop_gamma (float): Upper bound for gamma (cost) parameter range
		num_points_gamma (int): Number of gamma values to sample in the range
		start_beta (float): Lower bound for beta (mixer) parameter range
		stop_beta (float): Upper bound for beta (mixer) parameter range
		num_points_beta (int): Number of beta values to sample in the range
		heuristic (Callable[[np.ndarray], float]): Function that evaluates parameter quality.
			Takes parameter array of shape (2, 1) and returns scalar cost/energy
		plot (bool, optional): Whether to create 3D visualization. Default: True
		above (bool, optional): Whether to show top-down view of parameter space. Default: False
		save (bool, optional): Whether to save plots to files. Default: False
		matplot_save (bool, optional): Whether to save data in MATLAB format. Default: False
		
	Returns:
		Tuple[np.ndarray, np.ndarray, Tuple[int, int], Optional[plt.Figure]]: A tuple containing:
			- best_params (np.ndarray): Optimal parameters as [[gamma], [beta]] array
			- Z (np.ndarray): 2D grid of heuristic values for all parameter combinations
			- i (Tuple[int, int]): Grid indices (gamma_idx, beta_idx) of optimal parameters
			- fig (Optional[plt.Figure]): Matplotlib figure object if plot=True, else None
			
	Example:
		>>> def cost_function(params):
		...     return evaluate_qaoa_cost(params)
		>>> best_params, Z, indices, fig = grid_search(-π, π, 10, 0, π/2, 10, cost_function)
		>>> print(f"Optimal gamma: {best_params[0,0]:.3f}, beta: {best_params[1,0]:.3f}")
		
	Note:
		- Uses minimum-finding optimization (assumes heuristic returns cost to minimize)
		- Total function evaluations: num_points_gamma × num_points_beta
		- For p>1 circuits, use vec_grid_search_p2() or similar multi-layer methods
		- MATLAB export includes parameter ranges for post-processing
	"""
	# Create parameter grids
	X, Y, batch_array = get_batch_array(start_gamma, stop_gamma, num_points_gamma,
										start_beta, stop_beta, num_points_beta)
	
	# Evaluate heuristic for all parameter combinations
	Z = np.zeros((num_points_gamma, num_points_beta))
	
	for idx, params in enumerate(batch_array):
		gamma_idx = idx // num_points_beta
		beta_idx = idx % num_points_beta
		Z[gamma_idx, beta_idx] = heuristic(params)
	
	# Find best parameters
	i = np.unravel_index(Z.argmin(), Z.shape)
	
	fig = None
	if plot:
		fig = plot_grid_search(X, Y, Z, i, above=above, save=save)
	
	if matplot_save:
		mdic = {'Z_av': Z,
				'X': np.linspace(start_gamma, stop_gamma, num_points_gamma),
				'Y': np.linspace(start_beta, stop_beta, num_points_beta)}
		now = datetime.now()
		date_time = now.strftime("%m_%d_%Y")
		savemat(f'Matlab_{date_time}_{num_points_gamma}.mat', mdic)
	
	gamma = float(X[i[0]])
	beta = float(Y[i[1]])
	
	return np.array([[gamma], [beta]]), Z, i, fig

def get_batch_array(start_gamma: float, stop_gamma: float, num_points_gamma: int,
					start_beta: float, stop_beta: float, num_points_beta: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Create batch array for grid search over QAOA parameters.
	
	Generates all combinations of gamma (cost) and beta (mixer) parameters
	in the specified ranges for systematic grid search optimization. The output
	format is optimized for vectorized evaluation of QAOA circuits.
	
	Args:
		start_gamma (float): Lower bound for gamma (cost) parameter range
		stop_gamma (float): Upper bound for gamma (cost) parameter range
		num_points_gamma (int): Number of gamma values to sample uniformly in range
		start_beta (float): Lower bound for beta (mixer) parameter range
		stop_beta (float): Upper bound for beta (mixer) parameter range  
		num_points_beta (int): Number of beta values to sample uniformly in range
		
	Returns:
		Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
			- X (np.ndarray): 1D array of gamma parameter values, shape (num_points_gamma,)
			- Y (np.ndarray): 1D array of beta parameter values, shape (num_points_beta,)
			- batch_array (np.ndarray): All parameter combinations formatted for QAOA p=1,
			  shape (num_points_gamma × num_points_beta, 2, 1) where each element is
			  [[gamma], [beta]] for a single parameter combination
			  
	Example:
		>>> X, Y, batch = get_batch_array(-π, π, 3, 0, π/2, 2)
		>>> print(X.shape)  # (3,) - gamma values
		>>> print(Y.shape)  # (2,) - beta values  
		>>> print(batch.shape)  # (6, 2, 1) - 6 parameter combinations
		>>> print(batch[0])  # [[gamma_0], [beta_0]]
		
	Note:
		- Output batch_array is shaped for direct use with QAOA p=1 circuits
		- Uses numpy.linspace for uniform parameter sampling
		- Each batch element represents one complete set of QAOA parameters
		- Memory usage scales as O(num_points_gamma × num_points_beta)
	"""
	X = np.linspace(start_gamma, stop_gamma, num_points_gamma)
	Y = np.linspace(start_beta, stop_beta, num_points_beta)
	
	batch_list = []
	for x in X:
		for y in Y:
			temp = np.array([[float(x)], [float(y)]], dtype=float)
			batch_list.append(temp)
	
	batch_array = np.array(batch_list, dtype=float)
	return X, Y, batch_array

def plot_grid_search(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, i: Tuple[int, int], 
					 above: bool = False, name: str = '', save: bool = False, fontsize: int = 13) -> plt.Figure:
	"""
	Plot 3D visualization of grid search results.
	
	Creates a 3D surface plot showing how the objective function varies
	across the parameter space, with the optimal point highlighted. This
	visualization helps understand the optimization landscape and parameter
	sensitivity for QAOA circuits.
	
	Args:
		X (np.ndarray): Gamma parameter values (cost parameters), shape (num_gamma,)
		Y (np.ndarray): Beta parameter values (mixer parameters), shape (num_beta,)
		Z (np.ndarray): Objective function values at each (gamma, beta) point,
						shape (num_gamma, num_beta)
		i (Tuple[int, int]): Index of optimal parameters (gamma_idx, beta_idx)
		above (bool, optional): Whether to show top-down view. Default: False
		name (str, optional): Label for z-axis and filename. Default: ''
		save (bool, optional): Whether to save plot to file. Default: False
		fontsize (int, optional): Font size for labels and title. Default: 13
		
	Returns:
		plt.Figure: The figure object containing the 3D plot
		
	Features:
		- 3D surface plot with color mapping showing parameter landscape
		- Red star marking the optimal parameter combination
		- Configurable viewing angle (3D perspective or top-down)
		- Automatic axis formatting and legend
		- Optional file saving in PDF format
		
	Example:
		>>> fig = plot_grid_search(gamma_vals, beta_vals, cost_grid, (5, 3))
		>>> plt.show()  # Display the interactive 3D plot
		
	Note:
		- Uses BrBG colormap for good visibility of landscape features
		- Optimal point is highlighted for easy identification
		- Top-down view (above=True) useful for contour-like analysis
		- Figure numbers are randomized to avoid conflicts in notebooks
	"""
	# Create figure with better error handling
	try:
		fig = plt.figure(np.random.randint(51, 60), figsize=(12, 8))
		ax = fig.add_subplot(projection="3d")
		
		# Verify we have valid data to plot
		if len(X) == 0 or len(Y) == 0 or Z.size == 0:
			print("Warning: Empty data arrays provided to plot_grid_search")
			ax.text(0.5, 0.5, 0.5, 'No data to plot', ha='center', va='center', 
					transform=ax.transAxes, fontsize=16)
			return fig
		
		xx, yy = np.meshgrid(X, Y, indexing='ij')
		
		# Check if meshgrid and Z have compatible shapes
		if xx.shape != Z.shape or yy.shape != Z.shape:
			print(f"Warning: Shape mismatch - X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}")
			print(f"Meshgrid shapes - xx: {xx.shape}, yy: {yy.shape}")
			# Try to fix by reshaping Z if possible
			if Z.size == xx.size:
				Z = Z.reshape(xx.shape)
			else:
				ax.text(0.5, 0.5, 0.5, 'Data shape mismatch', ha='center', va='center',
						transform=ax.transAxes, fontsize=16)
				return fig
		
		surf = ax.plot_surface(xx, yy, Z, cmap=cm.BrBG, antialiased=False, alpha=0.8)
		
		# Add colorbar with error handling
		try:
			fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
		except Exception as cb_error:
			print(f"Could not add colorbar: {cb_error}")
		
		# Set axis properties with error handling
		try:
			ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
		except Exception:
			pass  # Continue without custom z-axis formatting
		
		# Plot optimal point
		if 0 <= i[0] < len(X) and 0 <= i[1] < len(Y):
			ax.plot(X[i[0]], Y[i[1]], Z[i], c="red", marker="*", 
					markersize=15, label="best params", zorder=10)
		
		# Set labels and title
		ax.set_xlabel("γ (cost parameter)", fontsize=fontsize)
		ax.set_ylabel("β (mixer parameter)", fontsize=fontsize)
		ax.set_zlabel(name if name else "Cost", fontsize=fontsize)
		
		plt.title(f'Best params: γ {X[i[0]]:.3f}, β {Y[i[1]]:.3f}', fontsize=fontsize)
		plt.legend(fontsize=fontsize)
		
		# Use tight_layout instead of constrained_layout to avoid warnings
		plt.tight_layout(pad=1.0)
		
		if save:
			plt.savefig(f'{name}_num_gamma{len(X)}.pdf', dpi=150, bbox_inches='tight')
		
		if above:
			ax.view_init(azim=0, elev=90)
			if save:
				plt.savefig(f'{name}_above_num_gamma{len(X)}.pdf', dpi=150, bbox_inches='tight')
		
		return fig
		
	except Exception as plot_error:
		print(f"Error in plot_grid_search: {plot_error}")
		# Create a simple fallback plot
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.text(0.5, 0.5, f'Plot failed: {str(plot_error)[:50]}...', 
				ha='center', va='center', transform=ax.transAxes, fontsize=12)
		ax.set_title('Grid Search Plot Error')
		return fig

def vec_grid_search_p2(start_gamma: float, stop_gamma: float, num_points_gamma: int,
						start_beta: float, stop_beta: float, num_points_beta: int,
						heuristic: Callable[[np.ndarray], float], vmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Perform grid search for QAOA p=2 circuits with optional JAX acceleration.
	
	Optimizes parameters for 2-layer QAOA circuits where each layer has independent
	gamma and beta parameters, resulting in a 4D parameter space. Supports optional
	JAX vectorization for significant speedup when available.
	
	Args:
		start_gamma (float): Lower bound for both gamma parameter ranges
		stop_gamma (float): Upper bound for both gamma parameter ranges
		num_points_gamma (int): Number of gamma values per layer (same for both layers)
		start_beta (float): Lower bound for both beta parameter ranges
		stop_beta (float): Upper bound for both beta parameter ranges
		num_points_beta (int): Number of beta values per layer (same for both layers)
		heuristic (Callable[[np.ndarray], float]): Evaluation function that takes
			parameter array of shape (2, 2) as [[gamma1, gamma2], [beta1, beta2]]
			and returns scalar objective value
		vmap (bool, optional): Whether to use JAX vectorization for acceleration. Default: False
		
	Returns:
		Tuple[np.ndarray, np.ndarray]: A tuple containing:
			- best_params (np.ndarray): Optimal parameters as [[gamma1, gamma2], [beta1, beta2]]
			- Z (np.ndarray): 4D grid of objective values, shape (num_gamma, num_gamma, num_beta, num_beta)
			
	Performance:
		- Without JAX: O(num_gamma² × num_beta²) sequential evaluations
		- With JAX: Vectorized evaluation with significant speedup for supported functions
		- Falls back gracefully to sequential evaluation if JAX fails
		
	Example:
		>>> def qaoa_p2_cost(params):
		...     return evaluate_2layer_qaoa(params)
		>>> best, Z = vec_grid_search_p2(-π, π, 5, 0, π/2, 5, qaoa_p2_cost, vmap=True)
		>>> print(f"Optimal: γ₁={best[0,0]:.3f}, γ₂={best[0,1]:.3f}")
		>>> print(f"         β₁={best[1,0]:.3f}, β₂={best[1,1]:.3f}")
		
	Note:
		- JAX acceleration requires compatible heuristic functions
		- Parameter ranges are identical for both layers (common in QAOA)
		- Memory usage: O(num_gamma² × num_beta²) for result storage
		- For p>2, similar patterns can be extended with higher-dimensional grids
	"""
	# gamma
	X1 = np.linspace(start_gamma, stop_gamma, num_points_gamma)
	X2 = np.linspace(start_gamma, stop_gamma, num_points_gamma)
	# beta
	Y1 = np.linspace(start_beta, stop_beta, num_points_beta)
	Y2 = np.linspace(start_beta, stop_beta, num_points_beta)

	if vmap:
		try:
			batch_list = []
			for x1 in X1:
				for x2 in X2:
					for y1 in Y1:
						for y2 in Y2:
							temp = np.array([[float(x1), float(x2)], [float(y1), float(y2)]])
							batch_list.append(temp)
			print('Batch list done!')

			batch_array = np.array(batch_list)
			import jax
			jax.config.update("jax_enable_x64", True)

			jit_circuit = jax.vmap(jax.jit(heuristic))
			Z = jit_circuit(batch_array)
			Z = Z.reshape(len(X1), len(X2), len(Y1), len(Y2))
			
		except Exception as e:
			print(f"JAX acceleration failed: {e}")
			print("Falling back to standard grid search")
			vmap = False

	if not vmap:
		Z = np.zeros((num_points_gamma, num_points_gamma, num_points_beta, num_points_beta))
		for xi, x1 in enumerate(X1):
			for xj, x2 in enumerate(X2):
				for yi, y1 in enumerate(Y1):
					for yj, y2 in enumerate(Y2):
						Z[xi, xj, yi, yj] = heuristic(np.array([[float(x1), float(x2)], [float(y1), float(y2)]]))

	# Find best Z
	i = np.unravel_index(Z.argmin(), Z.shape)
	
	gamma1 = float(X1[i[0]])
	gamma2 = float(X1[i[1]])
	beta1 = float(Y1[i[2]])
	beta2 = float(Y1[i[3]])

	return np.array([[gamma1, gamma2], [beta1, beta2]]), Z

def get_annealing_params(p: int, tau: float = 1.0, linear: bool = True, 
						sine: bool = False, plot: bool = False, save: bool = False) -> np.ndarray:
	"""
	Generate quantum annealing-inspired initial parameters for QAOA.
	
	Creates parameter schedules inspired by quantum annealing protocols,
	providing good starting points for QAOA optimization. The parameters
	follow annealing-like trajectories that smoothly interpolate between
	initial and final Hamiltonians.
	
	Args:
		p (int): Number of QAOA layers (circuit depth)
		tau (float, optional): Total annealing time parameter. Default: 1.0
		linear (bool, optional): Use linear annealing schedule. Default: True
		sine (bool, optional): Use sine-based annealing schedule. Default: False
		plot (bool, optional): Whether to plot parameter evolution. Default: False
		save (bool, optional): Whether to save parameters to file. Default: False
		
	Returns:
		np.ndarray: Parameter array of shape (2, p) where:
			- Row 0: gamma parameters (cost function angles)
			- Row 1: beta parameters (mixer function angles)
			- Columns: QAOA layers 1 through p
			
	Annealing Schedules:
		- Linear: B(s) = s, providing uniform parameter progression
		- Sine: B(s) = tan(-π/2 + s*π), with rapid transitions at boundaries
		
	Example:
		>>> params = get_annealing_params(p=3, tau=1.5, linear=True)
		>>> print(params.shape)  # (2, 3)
		>>> print(params[0])  # gamma parameters for 3 layers
		>>> print(params[1])  # beta parameters for 3 layers
		
	Theory:
		Based on Trotterized quantum annealing where the time-dependent
		Hamiltonian H(t) = (1-s(t))H_mixer + s(t)H_cost interpolates
		between mixer and cost Hamiltonians with s(t) ∈ [0,1].
		
	Note:
		- Linear schedule often works well for initial optimization attempts
		- Sine schedule provides more aggressive parameter changes near boundaries
		- Output can be used directly as starting point for gradient-based optimization
		- File saving uses comma-separated format compatible with most tools
	"""
	annealing_params = np.zeros((2, p))
	
	if linear:
		name = 'linear_'
		B_function = lambda s: s
		
		for i in range(p):
			s_mid = (i + 1 - 0.5) / p
			s_end = (i + 1 + 0.5) / p if i < p - 1 else 1.0
			
			annealing_params[0, i] = tau * B_function(s_mid)
			annealing_params[1, i] = -(tau / 2) * (
				2 - B_function(s_end) - B_function(s_mid))
		
		# Adjust last beta parameter
		annealing_params[1, p - 1] = -(tau / 2) * (1 - B_function((p - 0.5) / p))
	
	elif sine:
		name = 'sine_'
		B_function = lambda s: np.tan(-np.pi / 2 + s * np.pi)
		
		for i in range(p):
			s_mid = (i + 1 - 0.5) / p
			s_end = (i + 1 + 0.5) / p if i < p - 1 else 1.0
			
			annealing_params[0, i] = tau * B_function(s_mid)
			annealing_params[1, i] = -(tau / 2) * (
				2 - B_function(s_end) - B_function(s_mid))
		
		annealing_params[1, p - 1] = -(tau / 2) * (1 - B_function((p - 0.5) / p))
	
	if save:
		np.savetxt(f'{name}params.out', annealing_params, delimiter=',')
		if plot:
			plot_params(name, 1, p, save=True)
	
	return annealing_params

def interpolate_params(params: np.ndarray, only_last: bool = False, 
					   save: bool = False, plot: bool = False) -> np.ndarray:
	"""
	Interpolate QAOA parameters to increase circuit depth by one layer.
	
	Implements parameter interpolation strategy from Lucin et al. (Appendix B, p.14)
	to systematically extend optimized parameters from depth p to depth p+1.
	This allows leveraging optimization results from shallower circuits.
	
	Args:
		params (np.ndarray): Current parameter array of shape (2, p) where
			row 0 contains gamma parameters and row 1 contains beta parameters
		only_last (bool, optional): If True, only interpolate the last (new) parameter
			while keeping existing parameters unchanged. If False, interpolate all
			parameters. Default: False
		save (bool, optional): Whether to save interpolated parameters to file. Default: False
		plot (bool, optional): Whether to plot parameter evolution. Default: False
		
	Returns:
		np.ndarray: Interpolated parameters of shape (2, p+1) with the new layer
					parameters computed via weighted interpolation
					
	Interpolation Formula:
		For layer i in the expanded p+1 circuit:
		param_new[i] = ((i-1+1)/p) * param_old[i-1] + ((p-(i+1)+1)/p) * param_old[i]
		
	Example:
		>>> original_params = np.array([[0.5, 1.0], [0.3, 0.7]])  # 2-layer parameters
		>>> extended_params = interpolate_params(original_params)
		>>> print(extended_params.shape)  # (2, 3) - now 3 layers
		
	Use Cases:
		- Warm-starting optimization at higher depths
		- Progressive circuit depth increase in optimization
		- Parameter transfer between different problem sizes
		
	Note:
		- Based on theoretical analysis of adiabatic quantum computation
		- Provides smooth parameter transitions that preserve optimization structure
		- only_last=True useful for iterative depth expansion
		- Can be combined with further local optimization for best results
	"""
	p = params.shape[1]
	params_vector = np.concatenate((params, np.full((2, 1), 0.0)), axis=1)
	
	if only_last:
		i = p
		params_vector[0, i] = ((i - 1 + 1) / p) * params_vector[0, i - 1] + \
							  ((p - (i + 1) + 1) / p) * params_vector[0, i]
		params_vector[1, i] = ((i - 1 + 1) / p) * params_vector[1, i - 1] + \
							  ((p - (i + 1) + 1) / p) * params_vector[1, i]
		return_params = params_vector
	else:
		params_p_plus1 = np.full((2, p + 1), 0.0)
		for i in range(p + 1):
			params_p_plus1[0, i] = ((i - 1 + 1) / p) * params_vector[0, i - 1] + \
								   ((p - (i + 1) + 1) / p) * params_vector[0, i]
			params_p_plus1[1, i] = ((i - 1 + 1) / p) * params_vector[1, i - 1] + \
								   ((p - (i + 1) + 1) / p) * params_vector[1, i]
		return_params = params_p_plus1
	
	if save:
		name = 'interpolated'
		np.savetxt(f'{name}_params.out', return_params, delimiter=',')
		if plot:
			plot_params(name, 1, p + 1, save=True)
	
	return return_params

def plot_params(name: str, start_p: int, end_p: int, save: bool = False) -> None:
	"""
	Plot parameter evolution across different QAOA circuit depths.
	
	Visualizes how gamma and beta parameters change as the QAOA depth
	increases, useful for analyzing parameter interpolation strategies
	and understanding optimization landscapes across different depths.
	
	Args:
		name (str): Name identifier for the parameter set, used for file naming
			and plot titles. Should describe the parameter source (e.g., 'annealing',
			'interpolated', 'optimized')
		start_p (int): Starting QAOA depth for parameter evolution analysis
		end_p (int): Ending QAOA depth for parameter evolution analysis
		save (bool, optional): Whether to save the plot to file. Default: False
		
	Returns:
		None: Function creates and displays plots but returns nothing
		
	Expected Plot Features:
		- Separate traces for gamma and beta parameter evolution
		- X-axis: QAOA circuit depth (layer number)
		- Y-axis: Parameter values (angles)
		- Multiple lines showing how each parameter layer evolves
		- Legend identifying parameter types and layers
		
	File I/O:
		- Reads parameter files named '{name}_params.out' (if they exist)
		- Saves plots as '{name}_param_evolution.pdf' (if save=True)
		- Uses comma-separated format for parameter file parsing
		
	Note:
		This is a framework implementation. Full functionality would require:
		1. Parameter file reading from saved optimization results
		2. Matplotlib plotting setup with appropriate styling
		3. Error handling for missing or malformed parameter files
		4. Customizable plot appearance and output formats
		
	TODO:
		Complete implementation based on specific visualization requirements
		and available parameter file formats in the project.
	"""
	# Implementation for parameter plotting
	# This would be implemented based on specific plotting requirements
	pass
