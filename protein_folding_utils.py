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
import pennylane as qml

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

	def get_O_energies(self) -> List[float]:
		"""
		Extract one-body energy terms from the QUBO matrix.
		
		One-body terms correspond to diagonal elements of the QUBO matrix,
		representing local energy contributions for each qubit. These terms
		arise from penalty constraints and local field effects in the 
		optimization formulation.
		
		Returns:
			List[float]: One-body energy coefficients for each qubit, where
				  missing diagonal terms default to 0.0
				  
		Note:
			These energies are used to construct the σ^z terms in the 
			cost Hamiltonian: Σᵢ hᵢ σᵢ^z. In the HP lattice model, they
			primarily come from one-per constraints that ensure each
			amino acid occupies exactly one lattice site.
		"""
		O_energies = []
		for bit in self.bit_name_list:
			try:
				O_energies.append(self.Q[(bit, bit)])
			except KeyError:
				O_energies.append(0.0)
		return O_energies

	def get_T_energies(self) -> np.ndarray:
		"""
		Extract two-body energy terms from the QUBO matrix.
		
		Two-body terms represent interactions between different qubits,
		forming the off-diagonal elements of the QUBO matrix. These capture
		the coupling between different amino acid placements and include
		HP interactions, connectivity constraints, and overlap penalties.
		
		Returns:
			np.ndarray: Symmetric matrix of two-body interaction energies
						where T[i,j] represents the coupling between qubits i and j.
						Shape is (num_bits, num_bits) with T[i,j] = T[j,i].
						
		Note:
			The resulting matrix is symmetrized to ensure T[i,j] = T[j,i].
			These energies construct the σᵢ^z σⱼ^z terms in the cost Hamiltonian.
			In protein folding, they encode physical interactions like hydrophobic
			contacts (favorable) and constraint violations (penalized).
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

	def get_Dn(self) -> List[int]:
		"""
		Get cardinality constraints for XY-mixer implementation.
		
		Calculates the number of available lattice sites for even and odd
		amino acid positions, used for constructing XY-mixer Hamiltonians
		that preserve the Hamming weight within each subspace. This enforces
		the bipartite constraint structure of the HP lattice model.
		
		Returns:
			List[int]: Cardinality values for each amino acid position, where
				  even positions get ceiling(total_sites/2) sites and
				  odd positions get floor(total_sites/2) sites
				  
		Note:
			This enforces the constraint that even amino acids can only
			occupy even lattice sites and odd amino acids only odd sites,
			reducing the Hilbert space size by a factor of 2^(sequence_length).
			Essential for constructing constrained mixers in QAOA.
		"""
		D = []
		for seq in range(len(self.sequence)):
			if seq % 2 == 0:
				D.append(math.ceil((self.dim_lattice[0] * self.dim_lattice[1]) / 2))
			else:
				D.append(math.floor((self.dim_lattice[0] * self.dim_lattice[1]) / 2))
		return D

	def get_feasible_percentage(self) -> float:
		"""
		Calculate the percentage of solutions that are physically feasible.
		
		This metric provides insight into the constraint complexity of the
		protein folding problem. A lower percentage indicates tighter constraints
		and a more challenging optimization landscape, while higher percentages
		suggest more flexibility in solution space.
		
		Returns:
			float: Percentage of feasible solutions out of all possible 2^n states,
				   where n is the number of qubits (bits in the problem encoding)
			
		Example:
			If 4 out of 16 possible states are feasible, returns 25.0
			
		Note:
			This calculation requires both solution_set and feasible_set to be
			computed, which happens automatically during initialization. The
			ratio indicates how constrained the optimization problem is.
		"""
		return 100 * (len(self.feasible_set) / len(self.solution_set))

	def get_solution_set(self) -> List[np.ndarray]:
		"""
		Generate all possible binary configurations for the protein folding problem.
		
		Creates every possible assignment of amino acids to lattice positions,
		represented as binary strings of length num_bits. Each configuration
		corresponds to a computational basis state in the quantum optimization.
		If max_index is set, only returns the first max_index solutions to limit memory usage.
		
		Returns:
			List[np.ndarray]: List of numpy arrays, each representing a binary configuration
				  where 1 indicates an amino acid is placed at that position and 0 indicates
				  the position is empty. Each array has length num_bits.
				  
		Note:
			The total number of configurations is 2^num_bits, which can be
			very large (exponential scaling). Use max_index parameter to limit 
			computational cost for large systems. Each configuration represents
			one possible state in the quantum register.
		"""
		if self.max_index is not None:
			return [np.array(list(product([0, 1], repeat=self.num_bits)))[i] 
					for i in range(min(self.max_index, 2**self.num_bits))]
		return [np.array(i) for i in product([0, 1], repeat=self.num_bits)]

	def get_feasible_set(self) -> List[np.ndarray]:
		"""
		Get all feasible solutions that satisfy the protein folding constraints.
		
		Filters the complete solution space to identify only configurations that
		represent physically valid protein foldings. A solution is feasible if it
		satisfies all structural constraints of the HP lattice model.
		
		Returns:
			List[np.ndarray]: List of binary arrays representing feasible protein 
				configurations. Each array encodes a valid amino acid placement
				that satisfies all physical constraints.
		
		Feasibility Criteria:
			1. No overlaps: Same lattice node cannot contain multiple amino acids
			2. Connectivity: Adjacent amino acids in sequence must occupy neighboring 
			   lattice positions (Manhattan distance = 1)
			3. One-per constraint: Exactly one amino acid per sequence position
			4. Lattice placement: Even amino acids on even sites, odd on odd sites
			
		Algorithm:
			- Generates all possible amino acid placements using cardinality constraints
			- Tests each placement for overlap violations
			- Verifies connectivity constraints between consecutive amino acids
			- Returns only configurations passing all tests
			
		Note:
			The feasible set is typically much smaller than the full solution space,
			often representing <1% of total configurations for realistic proteins.
			This dramatic reduction reflects the stringent physical constraints.
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

	def manhattan_dist(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> int:
		"""
		Calculate Manhattan distance between two lattice nodes.
		
		The Manhattan distance is the sum of absolute differences between
		coordinates, representing the minimum number of lattice steps needed
		to move from one node to another along grid edges (no diagonal moves).
		
		Args:
			node1 (Tuple[int, int]): First lattice node coordinates (row, col)
			node2 (Tuple[int, int]): Second lattice node coordinates (row, col)
			
		Returns:
			int: Manhattan distance between the nodes
			
		Example:
			>>> manhattan_dist((0,0), (1,2)) returns 3 (|0-1| + |0-2|)
			>>> manhattan_dist((2,3), (2,4)) returns 1 (adjacent horizontally)
			>>> manhattan_dist((1,1), (1,1)) returns 0 (same position)
			
		Note:
			Used for connectivity constraint checking in protein folding.
			Adjacent amino acids must have Manhattan distance = 1.
		"""
		return sum(abs(a - b) for a, b in zip(node1, node2))

	def calc_solution_sets(self) -> None:
		"""
		Calculate and store both complete and feasible solution sets.
		
		This method computes all possible binary solutions and filters them
		to identify physically feasible protein configurations. The results
		are stored in instance variables for later use in optimization and
		analysis. This is useful for recomputing sets if problem parameters change.
		
		Updates:
			self.solution_set: All possible 2^n binary configurations
			self.feasible_set: Only configurations satisfying physical constraints
			
		Note:
			This method is typically called automatically during initialization,
			but can be used to recalculate sets if problem parameters change
			(e.g., lattice dimensions, sequence, or constraints).
		"""
		self.solution_set = self.get_solution_set()
		self.feasible_set = self.get_feasible_set()

	def viz_lattice(self, bit: np.ndarray) -> None:
		"""
		Visualize the protein configuration on the 2D lattice.
		
		Creates a plot showing the protein chain folded on the lattice, with
		amino acids colored by type (H/P) and connected by bonds. Each amino
		acid is labeled with its sequence index for easy identification.
		
		Args:
			bit (np.ndarray): Binary array representing the protein configuration
							  where 1 indicates amino acid placement at positions
							  defined by bit_name_list
							  
		Visualization Features:
			- Amino acids colored by type: H (hydrophobic) and P (polar) 
			- Chain connectivity shown with black lines between adjacent amino acids
			- Grid overlay for lattice position reference
			- Sequence indices displayed as white labels on each amino acid
			- Scatter plot with large markers for clear visibility
			
		Display Settings:
			- Figure size: 8×6 inches for good visibility
			- Colormap: 'coolwarm' for H/P distinction
			- Marker size: 1500 for clear amino acid visualization
			- Grid: Dotted lines for lattice reference
			
		Note:
			The function extracts amino acid positions from the binary representation
			and reconstructs the spatial protein configuration. The plot title includes
			the binary string for reference.
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
	
	plt.show()
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
	gamma2 = float(X2[i[1]])
	beta1 = float(Y1[i[2]])
	beta2 = float(Y2[i[3]])  # Fixed: was Y1[i[3]], should be Y2[i[3]]

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
