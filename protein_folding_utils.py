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

import numpy as np
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

class CoordinateBased_HPLattice:
	"""
	A coordinate-based HP lattice model for protein folding.
	
	This class implements the HP (Hydrophobic-Polar) lattice model where proteins
	are represented as chains of amino acids on a 2D lattice. Each amino acid is
	classified as either Hydrophobic (H=1) or Polar (P=0).
	"""
	
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
		self.Q = self.get_QUBO_matrix(verbose=verbose)
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

	def get_QUBO_matrix(self, verbose: bool = False) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
		"""
		Generate the QUBO (Quadratic Unconstrained Binary Optimization) matrix.
		
		The QUBO formulation includes:
		- HP interactions (maximize H-H contacts)
		- One-per constraints (each position occupied by exactly one amino acid)
		- Self-avoidance (no overlaps)
		- Connectivity (consecutive amino acids must be adjacent)
		
		Args:
			verbose (bool): If True, prints debugging information about constraint counts. Default: False
			
		Returns:
			Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]: QUBO matrix as dictionary where
				keys are tuples of ((node1, seq1), (node2, seq2)) and values are energy coefficients
		"""
		from collections import defaultdict
		
		Q = defaultdict(float)
		(L1, L2) = self.dim_lattice
		G = nx.grid_2d_graph(L1, L2)
		
		# Count for debugging
		count_HP = count_onper = count_sa = count_con = 0
		
		# HP interaction term (negative to favor H-H contacts)
		for u in G.nodes():
			for v in G.neighbors(u):
				for i in range(len(self.sequence)):
					for j in range(len(self.sequence)):
						if (self.sequence[i] == 1 and self.sequence[j] == 1 and 
							abs(i - j) != 1):  # Not consecutive amino acids
							Q[((u, i), (v, j))] -= 1
							count_HP += 1
		
		# One-per constraints
		cates_even = [x for x in range(len(self.sequence)) if x % 2 == 0]
		cates_odd = [x for x in range(len(self.sequence)) if x % 2 != 0]
		
		# Even positions
		for i in cates_even:
			for u in G.nodes():
				if sum(u) % 2 != 1:  # Even lattice sites
					Q[((u, i), (u, i))] -= 1 * self.lambda_vector[0]
					count_onper += 1
			
			for u in G.nodes():
				for v in G.nodes():
					if (u != v and sum(u) % 2 != 1 and sum(v) % 2 != 1):
						Q[((u, i), (v, i))] += 2 * self.lambda_vector[0]
						count_onper += 1
		
		# Odd positions
		for i in cates_odd:
			for u in G.nodes():
				if sum(u) % 2 == 1:  # Odd lattice sites
					Q[((u, i), (u, i))] -= 1 * self.lambda_vector[0]
					count_onper += 1
			
			for u in G.nodes():
				for v in G.nodes():
					if (u != v and sum(u) % 2 == 1 and sum(v) % 2 == 1):
						Q[((u, i), (v, i))] += 2 * self.lambda_vector[0]
						count_onper += 1
		
		# Self-avoidance constraints
		for u in G.nodes():
			if sum(u) % 2 != 1:  # Even sites
				for x in cates_even:
					for y in cates_even:
						if x != y and x < y:
							Q[((u, x), (u, y))] += 1 * self.lambda_vector[1]
							count_sa += 1
			elif sum(u) % 2 == 1:  # Odd sites
				for x in cates_odd:
					for y in cates_odd:
						if x != y and x < y:
							Q[((u, x), (u, y))] += 1 * self.lambda_vector[1]
							count_sa += 1
		
		# Connectivity constraints
		for i in cates_even:
			for u in G.nodes():
				for v in G.nodes():
					if (((u, v) not in G.edges()) and ((v, u) not in G.edges()) and u != v):
						if (sum(u) % 2 != 1 and sum(v) % 2 == 1):
							if len(self.sequence) % 2 == 0:
								Q[((u, i), (v, i + 1))] += 1 * self.lambda_vector[2]
								count_con += 1
							elif len(self.sequence) % 2 == 1 and i != cates_even[-1]:
								Q[((u, i), (v, i + 1))] += 1 * self.lambda_vector[2]
								count_con += 1
		
		for i in cates_odd:
			for u in G.nodes():
				for v in G.nodes():
					if (((u, v) not in G.edges()) and ((v, u) not in G.edges()) and u != v):
						if (sum(u) % 2 != 1 and sum(v) % 2 == 1):
							if len(self.sequence) % 2 == 1:
								Q[((u, i + 1), (v, i))] += 1 * self.lambda_vector[2]
								count_con += 1
							elif len(self.sequence) % 2 == 0 and i != cates_odd[-1]:
								Q[((u, i + 1), (v, i))] += 1 * self.lambda_vector[2]
								count_con += 1
		
		if verbose:
			print('QUBO matrix construction counts:')
			print(f'HP interactions: {count_HP}')
			print(f'One-per constraints (lambda 1): {count_onper}')
			print(f'Self-avoidance (lambda 2): {count_sa}')
			print(f'Connectivity (lambda 3): {count_con}')
		
		return dict(Q)  # Convert to regular dict

	def get_node_list(self, verbose: bool = False) -> List[Tuple[int, int]]:
		"""
		Get the list of lattice nodes in snake-like (serpentine) order.
		
		Creates an ordering of lattice nodes that follows a serpentine pattern:
		even rows go left-to-right, odd rows go right-to-left. This ordering
		is used to establish a canonical mapping between lattice positions
		and qubit indices.
		
		Args:
			verbose (bool): If True, saves a visualization of the lattice
							and prints the node ordering. Default: False
							
		Returns:
			List[Tuple[int, int]]: Ordered list of (row, col) tuples representing lattice nodes
								   in serpentine order
								  
		Note:
			The serpentine ordering helps maintain spatial locality in the
			qubit mapping, which can be beneficial for quantum circuit implementation.
		"""
		node_list = []
		(Lrow, Lcol) = self.dim_lattice
		G = nx.grid_2d_graph(Lrow, Lcol)
		
		for row in range(Lrow):
			start_index = row * Lcol
			if row % 2 == 0:  # Even row: forward
				node_list.extend(list(G.nodes())[start_index:start_index + Lcol])
			else:  # Odd row: backward
				node_list.extend(reversed(list(G.nodes())[start_index:start_index + Lcol]))
		
		if verbose:
			nx.draw(G, with_labels=True)
			plt.savefig('Lattice.png')
			print(f"Node list: {node_list}")
		
		return node_list

	def get_bit_names(self) -> List[Tuple[Tuple[int, int], int]]:
		"""
		Generate bit names mapping qubits to (lattice_node, amino_acid_index) pairs.
		
		Creates a systematic mapping between qubit indices and protein folding
		variables. Each bit represents placing a specific amino acid at a specific
		lattice position, following the constraint that even amino acids can only
		occupy even lattice sites and odd amino acids only odd sites.
		
		Returns:
			List[Tuple[Tuple[int, int], int]]: List of tuples where each element is 
				((row, col), amino_acid_index) representing the mapping from qubits 
				to protein folding variables
				  
		Note:
			The ordering respects the even/odd constraints: bits for even amino acids
			are assigned to even lattice sites first, then bits for odd amino acids
			to odd lattice sites. This reduces the problem's Hilbert space.
		"""
		seq_index = range(len(self.sequence))
		node_list = self.get_node_list(verbose=False)
		bit_name_list = []
		L_2 = int(self.dim_lattice[0] * self.dim_lattice[1])
		
		nodes_even = [x for x in range(L_2) if x % 2 == 0]
		nodes_odd = [x for x in range(L_2) if x % 2 != 0]
		
		for f in seq_index:
			if f % 2 == 0:  # Even sequence positions
				for s in nodes_even:
					bit_name_list.append((node_list[s], f))
			else:  # Odd sequence positions
				for s in nodes_odd:
					bit_name_list.append((node_list[s], f))
		
		return bit_name_list

	def get_O_energies(self):
		"""
		Extract one-body energy terms from the QUBO matrix.
		
		One-body terms correspond to diagonal elements of the QUBO matrix,
		representing local energy contributions for each qubit.
		
		Returns:
			list: One-body energy coefficients for each qubit, where
				  missing diagonal terms default to 0.0
				  
		Note:
			These energies are used to construct the σ^z terms in the 
			cost Hamiltonian: Σᵢ hᵢ σᵢ^z
		"""
		O_energies = []
		for bit in self.bit_name_list:
			try:
				O_energies.append(self.Q[(bit, bit)])
			except KeyError:
				O_energies.append(0.0)
		return O_energies

	def get_T_energies(self):
		"""
		Extract two-body energy terms from the QUBO matrix.
		
		Two-body terms represent interactions between different qubits,
		forming the off-diagonal elements of the QUBO matrix.
		
		Returns:
			np.ndarray: Symmetric matrix of two-body interaction energies
						where T[i,j] represents the coupling between qubits i and j
						
		Note:
			The resulting matrix is symmetrized to ensure T[i,j] = T[j,i].
			These energies construct the σᵢ^z σⱼ^z terms in the cost Hamiltonian.
		"""
		T_energies = np.zeros((self.num_bits, self.num_bits))
		
		for j in range(self.num_bits):
			for k in range(self.num_bits):
				if j != k:
					try:
						energy = self.Q.get((self.bit_name_list[j], self.bit_name_list[k]), 0.0)
						T_energies[j, k] = energy
					except KeyError:
						pass
		
		# Make symmetric
		T_energies = np.triu(T_energies)
		T_energies = T_energies + T_energies.T - np.diag(np.diag(T_energies))
		return T_energies

	def get_Dn(self):
		"""
		Get cardinality constraints for XY-mixer implementation.
		
		Calculates the number of available lattice sites for even and odd
		amino acid positions, used for constructing XY-mixer Hamiltonians
		that preserve the Hamming weight within each subspace.
		
		Returns:
			list: Cardinality values for each amino acid position, where
				  even positions get ceiling(total_sites/2) sites and
				  odd positions get floor(total_sites/2) sites
				  
		Note:
			This enforces the constraint that even amino acids can only
			occupy even lattice sites and odd amino acids only odd sites,
			reducing the Hilbert space size by a factor of 2.
		"""
		D = []
		for seq in range(len(self.sequence)):
			if seq % 2 == 0:
				D.append(math.ceil((self.dim_lattice[0] * self.dim_lattice[1]) / 2))
			else:
				D.append(math.floor((self.dim_lattice[0] * self.dim_lattice[1]) / 2))
		return D

	def get_feasible_percentage(self):
		"""
		Calculate the percentage of solutions that are physically feasible.
		
		This metric provides insight into the constraint complexity of the
		protein folding problem. A lower percentage indicates tighter constraints
		and a more challenging optimization landscape.
		
		Returns:
			float: Percentage of feasible solutions out of all possible 2^n states
			
		Example:
			If 4 out of 16 possible states are feasible, returns 25.0
			
		Note:
			This calculation requires both solution_set and feasible_set to be
			computed, which happens automatically during initialization.
		"""
		return 100 * (len(self.feasible_set) / len(self.solution_set))

	def get_solution_set(self):
		"""
		Generate all possible binary configurations for the protein folding problem.
		
		Creates every possible assignment of amino acids to lattice positions,
		represented as binary strings of length num_bits. If max_index is set,
		only returns the first max_index solutions to limit memory usage.
		
		Returns:
			list: List of numpy arrays, each representing a binary configuration
				  where 1 indicates an amino acid is placed at that position
				  
		Note:
			The total number of configurations is 2^num_bits, which can be
			very large. Use max_index parameter to limit computational cost
			for large systems.
		"""
		if self.max_index is not None:
			return [np.array(list(product([0, 1], repeat=self.num_bits)))[i] 
					for i in range(min(self.max_index, 2**self.num_bits))]
		return [np.array(i) for i in product([0, 1], repeat=self.num_bits)]

	def get_feasible_set(self):
		"""
		Get all feasible solutions that satisfy the problem constraints.
		
		A solution is feasible if:
		1. No overlaps (same node, different amino acids)
		2. Adjacent amino acids are neighbors on the lattice
		"""
		feasible_list = []
		index_list = []
		start = 0
		
		# Create index groups for each amino acid position
		for rot in self.Dn:
			stop = start + rot
			index_perm = list(range(start, stop))
			index_list.append(index_perm)
			start = stop
		
		# Generate all combinations
		comb = list(product(*index_list))
		
		for i in comb:
			state = np.zeros(self.num_bits)
			for j in i:
				state[j] = 1
			
			feasible = True
			
			# Check for overlaps (same node, different amino acids)
			for b in range(self.num_bits):
				node1 = self.bit_name_list[b][0]
				next_idx = (b + self.dim_lattice[0] * self.dim_lattice[1]) % self.num_bits
				node2 = self.bit_name_list[next_idx][0]
				
				if (node1 == node2 and state[b] and state[next_idx]):
					feasible = False
					break
			
			# Check connectivity (adjacent amino acids must be neighbors)
			if feasible:
				active_bits = [idx for idx, val in enumerate(state) if val == 1]
				for bit1_idx, bit1 in enumerate(active_bits[:-1]):
					bit2 = active_bits[bit1_idx + 1]
					node1 = self.bit_name_list[bit1][0]
					node2 = self.bit_name_list[bit2][0]
					
					if self.manhattan_dist(node1, node2) > 1:
						feasible = False
						break
			
			if feasible:
				feasible_list.append(state)
		
		return feasible_list

	def manhattan_dist(self, node1, node2):
		"""
		Calculate Manhattan distance between two lattice nodes.
		
		The Manhattan distance is the sum of absolute differences between
		coordinates, representing the minimum number of lattice steps needed
		to move from one node to another.
		
		Args:
			node1 (tuple): First lattice node coordinates (row, col)
			node2 (tuple): Second lattice node coordinates (row, col)
			
		Returns:
			int: Manhattan distance between the nodes
			
		Example:
			manhattan_dist((0,0), (1,2)) returns 3 (|0-1| + |0-2|)
		"""
		return sum(abs(a - b) for a, b in zip(node1, node2))

	def calc_solution_sets(self):
		"""
		Calculate and store both complete and feasible solution sets.
		
		This method computes all possible binary solutions and filters them
		to identify physically feasible protein configurations. The results
		are stored in instance variables for later use.
		
		Updates:
			self.solution_set: All possible 2^n binary configurations
			self.feasible_set: Only configurations satisfying physical constraints
			
		Note:
			This method is typically called automatically during initialization,
			but can be used to recalculate sets if problem parameters change.
		"""
		self.solution_set = self.get_solution_set()
		self.feasible_set = self.get_feasible_set()

	def viz_lattice(self, bit):
		"""
		Visualize the protein configuration on the 2D lattice.
		
		Creates a plot showing the protein chain folded on the lattice, with
		amino acids colored by type (H/P) and connected by bonds. Each amino
		acid is labeled with its sequence index.
		
		Args:
			bit (np.ndarray): Binary array representing the protein configuration
							  where 1 indicates amino acid placement at positions
							  defined by bit_name_list
							  
		Displays:
			- Matplotlib figure showing the folded protein
			- Hydrophobic (H=1) and Polar (P=0) amino acids in different colors
			- Chain connectivity with black lines
			- Grid overlay for lattice visualization
			- Amino acid sequence indices as white labels
			
		Note:
			The function extracts amino acid positions from the binary representation
			and reconstructs the spatial protein configuration for visualization.
		"""
		x, y = [], []
		
		for i in range(len(self.sequence)):
			for j in range(len(bit)):
				if bit[j] == 1 and self.bit_name_list[j][1] == i:
					node = self.bit_name_list[j][0]
					x.append(node[0])
					y.append(node[1])
					break
		
		plt.figure(figsize=(8, 6))
		plt.plot(y, x, 'k-', zorder=0)  # Connect with lines
		plt.scatter(y, x, c=self.sequence, cmap='coolwarm', s=1500, zorder=3)
		
		plt.margins(0.2)
		plt.gca().set_aspect('equal')
		plt.axis('on')
		
		ax = plt.gca()
		ax.xaxis.set_major_locator(MultipleLocator(1))
		ax.xaxis.set_major_formatter(NullFormatter())
		ax.yaxis.set_major_locator(MultipleLocator(1))
		ax.yaxis.set_major_formatter(NullFormatter())
		ax.tick_params(axis='both', length=0)
		plt.grid(True, ls=':')
		plt.title(f'Protein Configuration: {bit}')
		
		# Add amino acid indices
		for i in range(len(self.sequence)):
			plt.annotate(i, (y[i], x[i]), color='white', fontsize=24, 
						weight='bold', ha='center')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sort_over_threshold(x, threshold):
	"""
	Sort and filter values above a threshold.
	
	Args:
		x (np.ndarray): Array of values to filter and sort
		threshold (float): Minimum threshold value for filtering
		
	Returns:
		tuple: (filtered_values, indices) where filtered_values are the 
			   values above threshold sorted in ascending order, and indices
			   are the corresponding original indices
	"""
	mask = x > threshold
	sort_idx = x.argsort()
	mask_idx = sort_idx[mask[sort_idx]]
	return x[mask_idx], mask_idx

def plot_probs_with_energy(probs, num_qubits, H_cost, ground_states_i, 
						   new_fig=True, save=False, name='', threshold=0.001):
	"""
	Plot probabilities with corresponding energies and bit strings.
	
	This function creates a bar chart showing the probability of measuring
	different quantum states, colored by their classical energy. Ground states
	are highlighted in green.
	
	Args:
		probs (np.ndarray): Array of measurement probabilities for each quantum state
		num_qubits (int): Number of qubits in the quantum system
		H_cost: Cost Hamiltonian for energy calculations
		ground_states_i (np.ndarray): Indices of ground states
		new_fig (bool): Whether to create a new figure. Default: True
		save (bool): Whether to save the plot to file. Default: False
		name (str): Name prefix for saved file. Default: ''
		threshold (float): Minimum probability threshold for display. Default: 0.001
		
	Returns:
		matplotlib.figure.Figure: The figure object containing the plot
		
	Note:
		States with probability below threshold are filtered out to avoid clutter.
		Ground states are marked with green x-axis labels.
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
	
	# Create figure (remove conflicting plt.figure call)
	fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
	
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
	
	plt.show()
	return fig

def energies_of_set(_set, H_cost, num_qubits):
	"""
	Calculate energies for a set of states.
	
	Args:
		_set: Either a set of bit arrays or indices representing quantum states
		H_cost: Cost Hamiltonian for energy calculations
		num_qubits (int): Number of qubits in the system
		
	Returns:
		np.ndarray: Array of energies corresponding to each state in the set
		
	Note:
		Function automatically detects whether input is bit arrays or indices
		and converts as needed.
	"""
	try:
		if len(_set[0]) >= num_qubits:
			indices = bit_array_set2indices(_set)
	except:
		indices = _set
	
	energies_index_states = get_energies_index_states(H_cost)
	energies = np.take(energies_index_states, indices)
	return energies

def get_energies_index_states(H_cost):
	"""
	Get energies for all computational basis states.
	
	Args:
		H_cost: Cost Hamiltonian operator
		
	Returns:
		np.ndarray: Real-valued energies for all 2^n computational basis states,
					where n is the number of qubits. Energies are rounded to 8 
					decimal places for numerical stability.
					
	Note:
		Uses the diagonal elements of the Hamiltonian's sparse matrix representation.
	"""
	matrix = H_cost.sparse_matrix()
	return matrix.diagonal().real.round(8)

def energy_of_index(index, H_cost):
	"""
	Get energy of a specific computational basis state.
	
	Args:
		index (int): Index of the computational basis state (0 to 2^n - 1)
		H_cost: Cost Hamiltonian operator
		
	Returns:
		float: Energy eigenvalue of the specified state
		
	Example:
		energy = energy_of_index(5, H_cost)  # Get energy of state |101⟩
		# Returns the energy eigenvalue for computational basis state 5
	"""
	energies = get_energies_index_states(H_cost)
	return energies[index]

def index_set2bit_string(indices, num_qubits):
	"""
	Convert computational basis state indices to bit string representations.
	
	Args:
		indices (list or np.ndarray): List of state indices to convert
		num_qubits (int): Number of qubits (determines bit string length)
		
	Returns:
		list: List of bit strings (e.g., ['000', '101', '110'])
		
	Example:
		index_set2bit_string([0, 5, 6], 3) returns ['000', '101', '110']
	"""
	bit_strings = []
	for index in indices:
		bit_string = format(index, f'0{num_qubits}b')
		bit_strings.append(bit_string)
	return bit_strings

def bit_array_set2indices(bit_arrays):
	"""
	Convert bit arrays to computational basis state indices.
	
	Args:
		bit_arrays (list): List of binary arrays representing quantum states
		
	Returns:
		list: List of corresponding integer indices
		
	Example:
		bit_array_set2indices([[0,0,1], [1,0,1]]) returns [1, 5]
		
	Note:
		Assumes bit arrays represent states in computational basis where
		leftmost bit is most significant (e.g., [1,0,1] = 5 in decimal).
	"""
	indices = []
	for bit_array in bit_arrays:
		index = int(''.join(map(str, bit_array.astype(int))), 2)
		indices.append(index)
	return indices

def get_ground_states_i(H_cost):
	"""
	Find all ground state indices for a given Hamiltonian.
	
	Args:
		H_cost: Cost Hamiltonian operator
		
	Returns:
		np.ndarray: Array of indices corresponding to ground states
					(states with minimum energy)
					
	Note:
		In degenerate cases, multiple states may have the same ground energy.
		This function returns all such states.
	"""
	energies = get_energies_index_states(H_cost)
	min_energy = np.min(energies)
	ground_indices = np.where(energies == min_energy)[0]
	return ground_indices

def get_ground_states_energy_and_indices(feasible_set, H_cost):
	"""
	Find ground state energy and indices from a feasible solution set.
	
	This function is specifically designed for protein folding problems where
	only a subset of quantum states represent physically valid protein configurations.
	
	Args:
		feasible_set (list): List of binary arrays representing feasible protein 
							configurations (states that satisfy physical constraints)
		H_cost: Cost Hamiltonian operator for energy evaluation
		
	Returns:
		tuple: (min_energy, ground_indices) where:
			- min_energy (float): Lowest energy among feasible states
			- ground_indices (np.ndarray): Indices in feasible_set that achieve min_energy
			
	Note:
		This differs from get_ground_states_i() in that it only considers
		physically feasible protein configurations, not all 2^n quantum states.
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

def grid_search(start_gamma, stop_gamma, num_points_gamma,
				start_beta, stop_beta, num_points_beta,
				heuristic, plot=True, above=False, save=False, 
				matplot_save=False):
	"""
	Perform grid search for optimal QAOA parameters.
	
	Args:
		start_gamma, stop_gamma: Range for gamma (cost) parameters
		num_points_gamma: Number of gamma points to test
		start_beta, stop_beta: Range for beta (mixer) parameters  
		num_points_beta: Number of beta points to test
		heuristic: Function to evaluate parameter quality
		plot: Whether to plot results
		above: Whether to show top view of 3D plot
		save: Whether to save plots
		matplot_save: Whether to save MATLAB format data
	
	Returns:
		best_params: Optimal [gamma, beta] parameters
		Z: Grid of heuristic values
		i: Index of best parameters
		fig: Matplotlib figure object (if plot=True, otherwise None)
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

def get_batch_array(start_gamma, stop_gamma, num_points_gamma,
					start_beta, stop_beta, num_points_beta):
	"""
	Create batch array for grid search over QAOA parameters.
	
	Generates all combinations of gamma (cost) and beta (mixer) parameters
	in the specified ranges for systematic grid search optimization.
	
	Args:
		start_gamma, stop_gamma (float): Range for gamma (cost) parameters
		num_points_gamma (int): Number of gamma values to sample
		start_beta, stop_beta (float): Range for beta (mixer) parameters  
		num_points_beta (int): Number of beta values to sample
		
	Returns:
		tuple: (X, Y, batch_array) where:
			- X (np.ndarray): Gamma parameter values
			- Y (np.ndarray): Beta parameter values  
			- batch_array (np.ndarray): All parameter combinations shaped as
			  (total_combinations, 2, 1) for QAOA p=1 circuits
			  
	Example:
		X, Y, batch = get_batch_array(-π, π, 3, 0, π/2, 2)
		# Returns 6 parameter combinations: 3 gamma × 2 beta values
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

def plot_grid_search(X, Y, Z, i, above=False, name='', save=False, fontsize=13):
	"""
	Plot 3D visualization of grid search results.
	
	Creates a 3D surface plot showing how the objective function varies
	across the parameter space, with the optimal point highlighted.
	
	Args:
		X (np.ndarray): Gamma parameter values (cost parameters)
		Y (np.ndarray): Beta parameter values (mixer parameters)
		Z (np.ndarray): Objective function values at each (gamma, beta) point
		i (tuple): Index of optimal parameters (gamma_idx, beta_idx)
		above (bool): Whether to show top-down view. Default: False
		name (str): Label for z-axis and filename. Default: ''
		save (bool): Whether to save plot to file. Default: False
		fontsize (int): Font size for labels and title. Default: 13
		
	Returns:
		matplotlib.figure.Figure: The figure object containing the 3D plot
		
	Note:
		The optimal point is marked with a red star. If above=True,
		shows bird's-eye view of the parameter landscape.
	"""
	fig = plt.figure(np.random.randint(51, 60), figsize=(12, 8), constrained_layout=True)
	ax = fig.add_subplot(projection="3d")
	
	xx, yy = np.meshgrid(X, Y, indexing='ij')
	surf = ax.plot_surface(xx, yy, Z, cmap=cm.BrBG, antialiased=False)
	
	ax.zaxis.set_label_coords(-1, 1)
	ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
	ax.plot(X[i[0]], Y[i[1]], Z[i], c="red", marker="*", 
			label="best params", zorder=10)
	
	plt.legend(fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.title(f'Best params: γ {X[i[0]]:.3f}, β {Y[i[1]]:.3f}')
	
	ax.set_xlabel("γ (cost parameter)", fontsize=fontsize)
	ax.set_ylabel("β (mixer parameter)", fontsize=fontsize)
	ax.set_zlabel(name, fontsize=fontsize)
	
	if save:
		plt.savefig(f'{name}_num_gamma{len(X)}.pdf')
	
	if above:
		ax.view_init(azim=0, elev=90)
		if save:
			plt.savefig(f'{name}_above_num_gamma{len(X)}.pdf')
	
	return fig

def vec_grid_search_p2(start_gamma, stop_gamma, num_points_gamma,
						start_beta, stop_beta, num_points_beta,
						heuristic, vmap=False):
	"""
	Calculates the best parameters [gamma, beta] for p=2, gives the lowest cost, within the interval given.
	If vmap is on then JAX and JIT is used for speeding up the calculations.
	
	Args:
		start_gamma, stop_gamma: Range for gamma parameters
		num_points_gamma: Number of gamma points
		start_beta, stop_beta: Range for beta parameters
		num_points_beta: Number of beta points
		heuristic: Evaluation function
		vmap: Whether to use JAX vectorization
	
	Returns:
		tuple: (best_params, Z) where best_params is optimal parameters array
			   and Z is the full results grid
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
	gamma2 = float(X2[i[1]])
	beta1 = float(Y1[i[2]])
	beta2 = float(Y2[i[3]])  # Fixed: was Y1[i[3]], should be Y2[i[3]]

	return np.array([[gamma1, gamma2], [beta1, beta2]]), Z

def get_annealing_params(p, tau=1.0, linear=True, sine=False, 
						plot=False, save=False):
	"""
	Generate quantum annealing-inspired initial parameters.
	
	Args:
		p: Number of QAOA layers
		tau: Total annealing time
		linear: Use linear annealing schedule
		sine: Use sine annealing schedule  
		plot: Whether to plot parameters
		save: Whether to save parameters
	
	Returns:
		annealing_params: Array of [gamma, beta] parameters
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

def interpolate_params(params, only_last=False, save=False, plot=False):
	"""
	Interpolate parameters to increase depth by 1.
	Based on Lucin appendix B, p 14.
	
	Args:
		params: Current parameter array
		only_last: Only interpolate the last parameter
		save: Whether to save results
		plot: Whether to plot results
	
	Returns:
		Interpolated parameters
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

def plot_params(name, start_p, end_p, save=False):
	"""
	Plot parameter evolution across QAOA depths.
	
	Visualizes how gamma and beta parameters change as the QAOA depth
	increases, useful for analyzing parameter interpolation strategies.
	
	Args:
		name (str): Name identifier for the parameter set
		start_p (int): Starting QAOA depth  
		end_p (int): Ending QAOA depth
		save (bool): Whether to save the plot. Default: False
		
	Returns:
		None: Displays parameter evolution plot
		
	Note:
		This is a placeholder implementation. Full implementation would
		read parameter files and create evolution plots showing how
		gamma and beta parameters scale with circuit depth.
		
	TODO:
		Implement full parameter plotting functionality based on
		specific visualization requirements.
	"""
	# Implementation for parameter plotting
	# This would be implemented based on specific plotting requirements
	pass
