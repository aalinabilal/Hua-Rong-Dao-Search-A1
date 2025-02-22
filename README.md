# Hua-Rong-Dao-Search-A1
This project provides solutions to the Hua Rang Dao puzzle using two search algorithms: Depth-First Search (DFS) and A* Search. The puzzle is a combinatorial challenge where the goal is to rearrange tiles within a grid to reach a target configuration.

# Problem Description:

The Hua Rang Dao puzzle is a sliding puzzle where the objective is to rearrange pieces within a grid to match a target configuration. The puzzle can be solved by exploring the state space using search algorithms, with DFS and A* Search implemented as the primary techniques to find a solution.

# Problem Details:

State Space: The possible configurations of the puzzle.
Goal State: The target configuration that the puzzle aims to reach.
Moves: A set of legal moves that can be performed on the puzzle.
Search Algorithms Implemented:
DFS: A depth-first exploration of the state space.
A* Search: A heuristic-based search method that uses a cost function to guide the search towards the goal state efficiently.

# Grid Layout
The grid is a 2D array of characters, where each character represents a part of a piece or an empty space. Here are the key representations:

- Main Piece (1): This represents the main piece (usually larger), which occupies multiple grid spaces. It is typically represented by the character 1.
- Single Pieces (2): These are smaller individual pieces that fit into the grid. They are represented by the character 2.
- 1x2 and 2x1 Pieces (<>, v, ^): These represent pieces that take up two adjacent grid cells. The character < or > denotes horizontal pieces (1x2), while v and ^ represent vertical pieces (2x1).
- Empty Space (.): Empty cells in the grid are denoted by . (dot), indicating there is no piece occupying that cell.
  
# Explanation of Search Algorithms

Depth-First Search (DFS)
DFS is an uninformed search algorithm that explores possible configurations deeply before backtracking. While it’s guaranteed to find a solution if one exists, it may not always find the optimal solution due to its deep exploration approach.

A* Search
A* is an informed search algorithm that uses a heuristic to guide its exploration towards the goal efficiently. The heuristic used here is the Manhattan distance, which calculates the sum of the absolute differences in row and column positions of each tile. A* guarantees the optimal solution as long as the heuristic is admissible.


# Usage
To run the puzzle solver, use the following command in your terminal:

`python hrd.py --inputfile [name_of_input_file.txt] --outputfile [name_of_output_file.txt] --algo [dfs|astar]`

inputfile: Provide the input file in .txt format that contains the initial configuration of the puzzle.
outputfile: Specify the name of the output file where the solution will be saved.
algo: Choose between dfs (Depth-First Search) or astar (A* Search) to solve the puzzle.

Example usage:
`python hrd.py --inputfile puzzle_input.txt --outputfile solution_output.txt --algo dfs`

This will solve the puzzle using the DFS algorithm, and the solution will be written to solution_output.txt

