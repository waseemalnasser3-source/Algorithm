"""
Genetic Operators Module
Implements selection, crossover, and mutation operations
Developed by: Student 2
"""

import numpy as np
import random
from .chromosome import Chromosome


class GeneticOperators:
    """
    GeneticOperators class implementing GA operations
    Selection, Crossover, and Mutation
    """
    
    def __init__(self, crossover_rate=0.8, mutation_rate=0.1):
        """
        Initialize Genetic Operators
        
        Args:
            crossover_rate: Probability of crossover (default: 0.8)
            mutation_rate: Probability of mutation per gene (default: 0.1)
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    # ==================== SELECTION OPERATORS ====================
    
    def tournament_selection(self, population, tournament_size=3):
        """
        Tournament selection
        Randomly select K chromosomes and return the best one
        
        Args:
            population: List of Chromosome objects
            tournament_size: Number of chromosomes in tournament
            
        Returns:
            Selected Chromosome
        """
        # Randomly select tournament_size chromosomes
        tournament = random.sample(population, tournament_size)
        
        # Return the best one
        winner = max(tournament, key=lambda c: c.fitness)
        return winner.copy()
    
    def roulette_wheel_selection(self, population):
        """
        Roulette wheel selection (fitness-proportionate selection)
        Probability of selection proportional to fitness
        
        Args:
            population: List of Chromosome objects
            
        Returns:
            Selected Chromosome
        """
        # Get all fitness values
        fitness_values = np.array([c.fitness for c in population])
        
        # Handle negative fitness (shift to positive)
        if fitness_values.min() < 0:
            fitness_values = fitness_values - fitness_values.min()
        
        # Calculate selection probabilities
        total_fitness = fitness_values.sum()
        if total_fitness == 0:
            # If all fitness values are 0, select randomly
            return random.choice(population).copy()
        
        probabilities = fitness_values / total_fitness
        
        # Select based on probabilities
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx].copy()
    
    def rank_selection(self, population):
        """
        Rank-based selection
        Selection probability based on rank, not raw fitness
        
        Args:
            population: List of Chromosome objects
            
        Returns:
            Selected Chromosome
        """
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda c: c.fitness)
        
        # Assign ranks (1 to N)
        ranks = np.arange(1, len(population) + 1)
        
        # Calculate selection probabilities based on ranks
        total_rank = ranks.sum()
        probabilities = ranks / total_rank
        
        # Select based on probabilities
        selected_idx = np.random.choice(len(population), p=probabilities)
        return sorted_pop[selected_idx].copy()
    
    def elitism_selection(self, population, n_elite):
        """
        Elitism: Select top N chromosomes directly
        
        Args:
            population: List of Chromosome objects
            n_elite: Number of elite chromosomes to select
            
        Returns:
            List of elite Chromosomes
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
        
        # Return top N
        return [c.copy() for c in sorted_pop[:n_elite]]
    
    # ==================== CROSSOVER OPERATORS ====================
    
    def single_point_crossover(self, parent1, parent2):
        """
        Single-point crossover
        Split at one random point and exchange segments
        
        Args:
            parent1, parent2: Parent Chromosome objects
            
        Returns:
            child1, child2: Two offspring Chromosomes
        """
        # Decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Select random crossover point
        n_genes = parent1.n_features
        crossover_point = random.randint(1, n_genes - 1)
        
        # Create offspring
        child1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])
        
        # Ensure at least one feature is selected
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    def two_point_crossover(self, parent1, parent2):
        """
        Two-point crossover
        Split at two random points and exchange middle segment
        
        Args:
            parent1, parent2: Parent Chromosome objects
            
        Returns:
            child1, child2: Two offspring Chromosomes
        """
        # Decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n_genes = parent1.n_features
        
        # Select two random crossover points
        point1 = random.randint(0, n_genes - 2)
        point2 = random.randint(point1 + 1, n_genes - 1)
        
        # Create offspring
        child1_genes = np.concatenate([
            parent1.genes[:point1],
            parent2.genes[point1:point2],
            parent1.genes[point2:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:point1],
            parent1.genes[point1:point2],
            parent2.genes[point2:]
        ])
        
        # Ensure at least one feature is selected
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    def uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover
        Each gene randomly chosen from either parent
        
        Args:
            parent1, parent2: Parent Chromosome objects
            
        Returns:
            child1, child2: Two offspring Chromosomes
        """
        # Decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n_genes = parent1.n_features
        
        # Create random mask
        mask = np.random.randint(0, 2, size=n_genes).astype(bool)
        
        # Create offspring using mask
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        
        # Ensure at least one feature is selected
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    # ==================== MUTATION OPERATORS ====================
    
    def bit_flip_mutation(self, chromosome):
        """
        Bit-flip mutation
        Each gene has mutation_rate probability of flipping
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated Chromosome
        """
        mutated = chromosome.copy()
        
        # For each gene
        for i in range(mutated.n_features):
            # Mutate with probability mutation_rate
            if random.random() < self.mutation_rate:
                mutated.flip_gene(i)
        
        # Ensure at least one feature is selected
        if mutated.genes.sum() == 0:
            random_idx = random.randint(0, mutated.n_features - 1)
            mutated.set_gene(random_idx, 1)
        
        return mutated
    
    def random_resetting_mutation(self, chromosome):
        """
        Random resetting mutation
        Randomly reset some genes to 0 or 1
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated Chromosome
        """
        mutated = chromosome.copy()
        
        # For each gene
        for i in range(mutated.n_features):
            # Mutate with probability mutation_rate
            if random.random() < self.mutation_rate:
                mutated.set_gene(i, random.randint(0, 1))
        
        # Ensure at least one feature is selected
        if mutated.genes.sum() == 0:
            random_idx = random.randint(0, mutated.n_features - 1)
            mutated.set_gene(random_idx, 1)
        
        return mutated
    
    def swap_mutation(self, chromosome):
        """
        Swap mutation
        Swap two randomly selected genes
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated Chromosome
        """
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Select two random positions
        idx1, idx2 = random.sample(range(mutated.n_features), 2)
        
        # Swap genes
        temp = mutated.genes[idx1]
        mutated.genes[idx1] = mutated.genes[idx2]
        mutated.genes[idx2] = temp
        
        mutated.n_selected_features = mutated.count_selected_features()
        
        return mutated


# Example usage
if __name__ == '__main__':
    print("="*60)
    print("GENETIC OPERATORS EXAMPLE")
    print("="*60)
    
    # Create operators
    operators = GeneticOperators(crossover_rate=0.8, mutation_rate=0.1)
    
    # Create two parent chromosomes
    n_features = 10
    parent1 = Chromosome(n_features, genes=[1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    parent2 = Chromosome(n_features, genes=[0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    
    # Assign some fitness values
    parent1.fitness = 0.85
    parent2.fitness = 0.75
    
    print("\nParents:")
    print(f"Parent 1: {parent1.genes} (fitness={parent1.fitness})")
    print(f"Parent 2: {parent2.genes} (fitness={parent2.fitness})")
    
    # Test crossover operators
    print("\n" + "="*60)
    print("CROSSOVER OPERATIONS")
    print("="*60)
    
    print("\nSingle-point crossover:")
    child1, child2 = operators.single_point_crossover(parent1, parent2)
    print(f"Child 1: {child1.genes}")
    print(f"Child 2: {child2.genes}")
    
    print("\nTwo-point crossover:")
    child1, child2 = operators.two_point_crossover(parent1, parent2)
    print(f"Child 1: {child1.genes}")
    print(f"Child 2: {child2.genes}")
    
    print("\nUniform crossover:")
    child1, child2 = operators.uniform_crossover(parent1, parent2)
    print(f"Child 1: {child1.genes}")
    print(f"Child 2: {child2.genes}")
    
    # Test mutation operators
    print("\n" + "="*60)
    print("MUTATION OPERATIONS")
    print("="*60)
    
    test_chromosome = Chromosome(n_features, genes=[1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    print(f"\nOriginal: {test_chromosome.genes}")
    
    mutated = operators.bit_flip_mutation(test_chromosome)
    print(f"After bit-flip mutation: {mutated.genes}")
    
    mutated = operators.random_resetting_mutation(test_chromosome)
    print(f"After random resetting: {mutated.genes}")
    
    # Test selection operators
    print("\n" + "="*60)
    print("SELECTION OPERATIONS")
    print("="*60)
    
    # Create a population
    population = Chromosome.create_random_population(5, n_features)
    for i, chrom in enumerate(population):
        chrom.fitness = np.random.random()  # Assign random fitness
        print(f"{i+1}. {chrom.genes} (fitness={chrom.fitness:.3f})")
    
    print("\nTournament selection (size=3):")
    selected = operators.tournament_selection(population, tournament_size=3)
    print(f"Selected: {selected.genes} (fitness={selected.fitness:.3f})")
    
    print("\nRoulette wheel selection:")
    selected = operators.roulette_wheel_selection(population)
    print(f"Selected: {selected.genes} (fitness={selected.fitness:.3f})")

