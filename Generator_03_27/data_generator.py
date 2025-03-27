from typing import List, Tuple, Union
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import random
import itertools

class ResolutionDataGenerator:
    def __init__(
        self,
        predicates: List[str] = None,
        variables: List[str] = None,
        constants: List[str] = None,
        functions: List[str] = None,
        max_clause_length: int = 3,
        max_term_arity: int = 3,
        max_function_arity: int = 1
    ):
        """Initialize the clause generator with customizable parameters.
        
        Args:
            predicates: List of predicate names to use
            variables: List of variable names to use
            constants: List of constant names to use
            functions: List of function symbols to use.
            max_clause_length: Maximum number of literals in a clause
            max_term_arity: Maximum number of terms in a predicate
            max_function_arity: Maximum number of arguments in a function.
        """
        self.predicates = predicates or ["P", "Q", "R", "S", "T"]
        self.variables = variables or ["x", "y", "z", "u", "v", "w"]
        self.constants = constants or ["a", "b", "c", "d", "e"]
        self.functions = functions or ["f", "g", "h"]
        self.max_clause_length = max_clause_length
        self.max_term_arity = max_term_arity
        self.max_function_arity = max_function_arity
    
    def generate_term(self, depth: int = 0) -> Union[str, Tuple[str, Tuple]]:
        """
        Generate a random term that can be a constant, variable, or function.
        
        Args:
            depth (int): Track recursion depth to prevent excessive nesting
        
        Returns:
            A term which can be:
            - A constant (str)
            - A variable (str)
            - A function with arguments respecting max_function_arity
        """
        # Prevent deep recursion and excessive function nesting
        if depth >= self.max_function_arity:
            return random.choice(self.constants + self.variables)
        
        # Randomly choose term type
        term_type = random.choice(['constant', 'variable', 'function'])
        
        if term_type == 'constant':
            return random.choice(self.constants)
        
        if term_type == 'variable':
            return random.choice(self.variables)
        
        # Function generation
        func = random.choice(self.functions)
        
        # Generate function arguments respecting max_function_arity
        args = tuple(
            self.generate_term(depth + 1) 
            for _ in range(random.randint(1, min(self.max_function_arity, 3)))
        )
        
        return (func, args)

    def generate_literal(self, allow_negation: bool = True) -> Tuple[str, Tuple, bool]:
        """
        Generate a random literal with support for functions.
        
        Args:
            allow_negation (bool): Whether to allow negated literals
        
        Returns:
            A tuple containing:
            - Predicate name (str)
            - Arguments (tuple of terms)
            - Negation flag (bool)
        """
        predicate = random.choice(self.predicates)
        
        # Determine arity of the predicate
        arity = random.randint(1, min(self.max_term_arity, 3))
        
        # Generate arguments with support for functions
        args = tuple(self.generate_term() for _ in range(arity))
        
        # Determine if the literal is negated
        is_negated = allow_negation and random.random() < 0.5
        
        return (predicate, args, is_negated)
    
    def generate_clause(self) -> List[Tuple[str, Tuple, bool]]:
        """
        Generate a random clause (a set of literals).
        
        Returns:
            A list of literals
        """
        # Determine number of literals in the clause
        num_literals = random.randint(1, self.max_clause_length)
        
        return [self.generate_literal() for _ in range(num_literals)]
    


    def are_complementary(self, literal1, literal2):
        """
        Check if two literals are complementary (can be resolved)
        
        Args:
            literal1 (tuple): First literal (predicate, args, is_negated)
            literal2 (tuple): Second literal (predicate, args, is_negated)
        
        Returns:
            bool: Whether literals are complementary
        """
        # Same predicate name
        if literal1[0] != literal2[0]:
            return False
        
        # Same arguments
        if literal1[1] != literal2[1]:
            return False
        
        # One negated, one not
        return literal1[2] != literal2[2]
    
    def generate_clause_graph(self, num_literals=5):
        """
        Generate a graph representing a clause with literals
        
        Args:
            num_literals (int): Number of literals in the clause
        
        Returns:
            torch_geometric.data.Data: Graph representation
        """
        # Generate literals
        literals = [self.generate_literal() for _ in range(num_literals)]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (literals)
        for i, literal in enumerate(literals):
            G.add_node(i, 
                       predicate=literal[0], 
                       args=literal[1], 
                       is_negated=int(literal[2]))
        
        # Add edges based on potential resolvability
        for (i, j) in itertools.combinations(range(len(literals)), 2):
            if self.are_complementary(literals[i], literals[j]):
                G.add_edge(i, j)
        
        # Convert to PyTorch Geometric Data
        x = torch.tensor([[
            self.predicates.index(node_data['predicate']),
            len(node_data['args']),
            node_data['is_negated']
        ] for _, node_data in G.nodes(data=True)], dtype=torch.float)
        
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def generate_dataset(self, num_graphs=1000):
        """
        Generate a dataset of clause graphs
        
        Args:
            num_graphs (int): Number of graphs to generate
        
        Returns:
            list: List of torch_geometric.data.Data objects
        """
        return [self.generate_clause_graph() for _ in range(num_graphs)]

# Example usage and demonstration
if __name__ == "__main__":
    # Create generator with custom max_function_arity
    generator = ResolutionDataGenerator(max_function_arity=2)
    
    print("Example Literals:")
    for _ in range(5):
        literal = generator.generate_literal()
        print(f"Literal: {literal}")
        
    print("\nExample Clause:")
    clause = generator.generate_clause()
    print("Clause:", clause)

# Example usage
# if __name__ == "__main__":
#     # Seed for reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     # Create data generator
#     generator = ResolutionDataGenerator()
    
#     # Generate dataset
#     dataset = generator.generate_dataset(num_graphs=100)
    
#     # Print some statistics
#     print(f"Generated {len(dataset)} graphs")
    
#     # Example of accessing a graph's properties
#     sample_graph = dataset[0]
#     print("\nSample Graph:")
#     print("Node Features:", sample_graph.x)
#     print("Edge Connections:", sample_graph.edge_index)
#     print("Number of Nodes:", sample_graph.num_nodes)
#     print("Number of Edges:", sample_graph.num_edges)
