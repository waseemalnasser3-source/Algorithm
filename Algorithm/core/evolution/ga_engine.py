"""
Evolutionary Optimization Core Engine
Bio-inspired optimization framework for intelligent attribute discovery
Developed by: Rania Hassan (Evolution Engine Developer)
"""

import numpy as np
import time
from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from .operators import GeneticOperators


class GeneticAlgorithm:
    """
    GeneticAlgorithm class - Main GA engine
    Orchestrates the entire genetic algorithm process
    """
    
    def __init__(self, 
                 population_size=50,
                 n_generations=100,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 elitism_rate=0.1,
                 tournament_size=3,
                 alpha=0.9,
                 model_type='random_forest',
                 selection_method='tournament',
                 crossover_method='single_point',
                 verbose=True):
        """
        Initialize Genetic Algorithm
        
        Args:
            population_size: Number of chromosomes in population
            n_generations: Number of generations to evolve
            crossover_rate: Probability of crossover (0.0 to 1.0)
            mutation_rate: Probability of mutation per gene (0.0 to 1.0)
            elitism_rate: Proportion of best chromosomes to keep (0.0 to 1.0)
            tournament_size: Size of tournament for tournament selection
            alpha: Weight for accuracy vs feature reduction (0.0 to 1.0)
            model_type: ML model ('random_forest', 'svm', 'knn')
            selection_method: Selection operator ('tournament', 'roulette', 'rank')
            crossover_method: Crossover operator ('single_point', 'two_point', 'uniform')
            verbose: Print progress information
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.model_type = model_type
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.verbose = verbose
        
        # Calculate number of elite chromosomes
        self.n_elite = max(1, int(population_size * elitism_rate))
        
        # Initialize components
        self.operators = GeneticOperators(crossover_rate, mutation_rate)
        self.evaluator = None
        
        # Track best solution and history
        self.best_chromosome = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'best_accuracy': [],
            'avg_features': []
        }
    
    def initialize_population(self, n_features):
        """
        Create initial random population
        
        Args:
            n_features: Number of features in dataset
            
        Returns:
            List of Chromosome objects
        """
        if self.verbose:
            print(f"Initializing population of {self.population_size} chromosomes...")
        
        population = Chromosome.create_random_population(
            self.population_size,
            n_features
        )
        
        return population
    
    def select_parent(self, population):
        """
        Select a parent using specified selection method
        
        Args:
            population: List of Chromosome objects
            
        Returns:
            Selected parent Chromosome
        """
        if self.selection_method == 'tournament':
            return self.operators.tournament_selection(population, self.tournament_size)
        elif self.selection_method == 'roulette':
            return self.operators.roulette_wheel_selection(population)
        elif self.selection_method == 'rank':
            return self.operators.rank_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover using specified method
        
        Args:
            parent1, parent2: Parent Chromosomes
            
        Returns:
            child1, child2: Offspring Chromosomes
        """
        if self.crossover_method == 'single_point':
            return self.operators.single_point_crossover(parent1, parent2)
        elif self.crossover_method == 'two_point':
            return self.operators.two_point_crossover(parent1, parent2)
        elif self.crossover_method == 'uniform':
            return self.operators.uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def mutate(self, chromosome):
        """
        Perform mutation on chromosome
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated Chromosome
        """
        return self.operators.bit_flip_mutation(chromosome)
    
    def evolve_generation(self, population):
        """
        Evolve one generation
        
        Args:
            population: Current population
            
        Returns:
            New population after evolution
        """
        # Select elite chromosomes
        elite = self.operators.elitism_selection(population, self.n_elite)
        
        # Create new population starting with elite
        new_population = elite.copy()
        
        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parent(population)
            parent2 = self.select_parent(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Ensure population size is exactly as specified
        new_population = new_population[:self.population_size]
        
        return new_population
    
    def update_history(self, population, generation):
        """
        Update history with current generation statistics
        
        Args:
            population: Current population
            generation: Current generation number
        """
        stats = self.evaluator.get_population_stats(population)
        
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['avg_fitness'].append(stats['avg_fitness'])
        self.history['worst_fitness'].append(stats['worst_fitness'])
        self.history['best_accuracy'].append(stats['best_accuracy'])
        self.history['avg_features'].append(stats['avg_features'])
        
        # Update best chromosome if better one found
        best = self.evaluator.get_best_chromosome(population)
        if self.best_chromosome is None or best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best.copy()
        
        # Print progress
        if self.verbose:
            print(f"Generation {generation:3d} | "
                  f"Best: {stats['best_fitness']:.4f} | "
                  f"Avg: {stats['avg_fitness']:.4f} | "
                  f"Acc: {stats['best_accuracy']:.4f} | "
                  f"Features: {stats['avg_features']:.1f}")
    
    def fit(self, X, y):
        """
        Run genetic algorithm on dataset
        
        Args:
            X: Feature matrix (samples x features)
            y: Target vector
            
        Returns:
            best_chromosome: Best solution found
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("GENETIC ALGORITHM FOR FEATURE SELECTION")
            print("="*70)
            print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Crossover rate: {self.crossover_rate}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Elite chromosomes: {self.n_elite}")
            print(f"Model: {self.model_type}")
            print("="*70)
        
        # Initialize fitness evaluator
        self.evaluator = FitnessEvaluator(
            X, y, 
            model_type=self.model_type,
            alpha=self.alpha
        )
        
        # Initialize population
        population = self.initialize_population(X.shape[1])
        
        # Evaluate initial population
        if self.verbose:
            print("\nEvaluating initial population...")
        self.evaluator.evaluate_population(population)
        self.update_history(population, 0)
        
        # Evolution loop
        if self.verbose:
            print(f"\n{'='*70}")
            print("Starting evolution...")
            print(f"{'='*70}\n")
        
        for generation in range(1, self.n_generations + 1):
            # Evolve new generation
            population = self.evolve_generation(population)
            
            # Evaluate new population
            self.evaluator.evaluate_population(population)
            
            # Update history
            self.update_history(population, generation)
        
        # Final results
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("EVOLUTION COMPLETED")
            print(f"{'='*70}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"\nBest Solution Found:")
            print(f"  Fitness: {self.best_chromosome.fitness:.4f}")
            print(f"  Accuracy: {self.best_chromosome.accuracy:.4f}")
            print(f"  Selected features: {self.best_chromosome.n_selected_features}/{X.shape[1]}")
            print(f"  Feature reduction: {(1 - self.best_chromosome.n_selected_features/X.shape[1])*100:.1f}%")
            print(f"  Selected indices: {self.best_chromosome.get_selected_features().tolist()}")
            print(f"{'='*70}\n")
        
        return self.best_chromosome
    
    def get_selected_features(self):
        """
        Get indices of features selected by best chromosome
        
        Returns:
            numpy array of feature indices
        """
        if self.best_chromosome is None:
            raise ValueError("GA has not been run yet. Call fit() first.")
        return self.best_chromosome.get_selected_features()
    
    def get_history(self):
        """
        Get evolution history
        
        Returns:
            dict: History of fitness values across generations
        """
        return self.history
    
    def plot_history(self):
        """
        Plot evolution history (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            
            generations = range(len(self.history['best_fitness']))
            
            plt.figure(figsize=(12, 4))
            
            # Plot fitness evolution
            plt.subplot(1, 2, 1)
            plt.plot(generations, self.history['best_fitness'], label='Best', linewidth=2)
            plt.plot(generations, self.history['avg_fitness'], label='Average', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot feature count
            plt.subplot(1, 2, 2)
            plt.plot(generations, self.history['avg_features'], linewidth=2, color='green')
            plt.xlabel('Generation')
            plt.ylabel('Average Features Selected')
            plt.title('Feature Selection Evolution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot history.")


# Example usage
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    # Load dataset
    X, y = load_iris(return_X_y=True)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=20,
        n_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_rate=0.1,
        alpha=0.9,
        model_type='random_forest',
        verbose=True
    )
    
    # Run GA
    best_solution = ga.fit(X, y)
    
    # Get selected features
    selected_features = ga.get_selected_features()
    print(f"Selected features: {selected_features}")
    
    # Plot history (if matplotlib available)
    # ga.plot_history()

