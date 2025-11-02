"""
Chromosome Module
Represents a solution (feature subset) in the genetic algorithm
Developed by: Student 2
"""

import numpy as np
import random


class Chromosome:
    """
    Chromosome class representing a binary feature selection solution
    Each gene (bit) represents whether a feature is selected (1) or not (0)
    """
    
    def __init__(self, n_features, genes=None):
        """
        Initialize a chromosome
        
        Args:
            n_features: Total number of features available
            genes: Binary array representing selected features (optional)
                  If None, generates random chromosome
        """
        self.n_features = n_features
        
        if genes is not None:
            self.genes = np.array(genes, dtype=int)
        else:
            # Generate random chromosome
            # Ensure at least one feature is selected
            self.genes = np.random.randint(0, 2, size=n_features)
            while self.genes.sum() == 0:  # No features selected
                self.genes = np.random.randint(0, 2, size=n_features)
        
        self.fitness = 0.0
        self.accuracy = 0.0
        self.n_selected_features = int(self.genes.sum())
    
    def get_selected_features(self):
        """
        Get indices of selected features
        
        Returns:
            numpy array of indices where genes = 1
        """
        return np.where(self.genes == 1)[0]
    
    def get_selected_feature_mask(self):
        """
        Get boolean mask of selected features
        
        Returns:
            Boolean array where True = selected feature
        """
        return self.genes.astype(bool)
    
    def count_selected_features(self):
        """
        Count number of selected features
        
        Returns:
            int: Number of features selected
        """
        return int(self.genes.sum())
    
    def flip_gene(self, index):
        """
        Flip a gene at given index (mutation operation)
        
        Args:
            index: Index of gene to flip
        """
        if 0 <= index < self.n_features:
            self.genes[index] = 1 - self.genes[index]
            self.n_selected_features = self.count_selected_features()
    
    def set_gene(self, index, value):
        """
        Set a gene to a specific value
        
        Args:
            index: Index of gene
            value: Value to set (0 or 1)
        """
        if 0 <= index < self.n_features and value in [0, 1]:
            self.genes[index] = value
            self.n_selected_features = self.count_selected_features()
    
    def copy(self):
        """
        Create a copy of this chromosome
        
        Returns:
            New Chromosome object with same genes
        """
        new_chromosome = Chromosome(self.n_features, genes=self.genes.copy())
        new_chromosome.fitness = self.fitness
        new_chromosome.accuracy = self.accuracy
        return new_chromosome
    
    def __str__(self):
        """
        String representation of chromosome
        
        Returns:
            String showing genes and fitness
        """
        genes_str = ''.join(map(str, self.genes))
        return f"Chromosome(genes={genes_str}, fitness={self.fitness:.4f}, features={self.n_selected_features}/{self.n_features})"
    
    def __repr__(self):
        """
        Representation for debugging
        """
        return self.__str__()
    
    def __lt__(self, other):
        """
        Less than comparison based on fitness
        Used for sorting chromosomes
        """
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        """
        Equality comparison based on genes
        """
        if not isinstance(other, Chromosome):
            return False
        return np.array_equal(self.genes, other.genes)
    
    def hamming_distance(self, other):
        """
        Calculate Hamming distance to another chromosome
        (Number of different genes)
        
        Args:
            other: Another Chromosome object
            
        Returns:
            int: Number of different genes
        """
        if not isinstance(other, Chromosome):
            raise TypeError("Can only compare with another Chromosome")
        
        return np.sum(self.genes != other.genes)
    
    @staticmethod
    def create_random_population(pop_size, n_features):
        """
        Create a population of random chromosomes
        
        Args:
            pop_size: Size of population to create
            n_features: Number of features
            
        Returns:
            List of Chromosome objects
        """
        population = []
        for _ in range(pop_size):
            chromosome = Chromosome(n_features)
            population.append(chromosome)
        return population
    
    @staticmethod
    def create_uniform_population(pop_size, n_features, selection_prob=0.5):
        """
        Create a population with uniform feature selection probability
        
        Args:
            pop_size: Size of population
            n_features: Number of features
            selection_prob: Probability of selecting each feature
            
        Returns:
            List of Chromosome objects
        """
        population = []
        for _ in range(pop_size):
            genes = (np.random.random(n_features) < selection_prob).astype(int)
            # Ensure at least one feature is selected
            if genes.sum() == 0:
                genes[random.randint(0, n_features-1)] = 1
            chromosome = Chromosome(n_features, genes=genes)
            population.append(chromosome)
        return population


# Example usage
if __name__ == '__main__':
    print("="*60)
    print("CHROMOSOME EXAMPLE")
    print("="*60)
    
    # Create a chromosome with 10 features
    n_features = 10
    chromosome = Chromosome(n_features)
    
    print(f"\nRandom Chromosome:")
    print(chromosome)
    print(f"Selected features: {chromosome.get_selected_features()}")
    
    # Create specific chromosome
    genes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    chromosome2 = Chromosome(n_features, genes=genes)
    print(f"\nSpecific Chromosome:")
    print(chromosome2)
    print(f"Selected features: {chromosome2.get_selected_features()}")
    
    # Flip a gene (mutation)
    print(f"\nBefore mutation: {chromosome2.genes}")
    chromosome2.flip_gene(1)
    print(f"After flipping gene 1: {chromosome2.genes}")
    
    # Calculate Hamming distance
    distance = chromosome.hamming_distance(chromosome2)
    print(f"\nHamming distance between chromosomes: {distance}")
    
    # Create random population
    population = Chromosome.create_random_population(5, n_features)
    print(f"\nRandom Population (size=5):")
    for i, chrom in enumerate(population):
        print(f"  {i+1}. {chrom}")

