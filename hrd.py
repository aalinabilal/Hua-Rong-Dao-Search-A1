from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys
import copy
from typing import List

# ====================================================================================

char_goal = '1'
char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def get_coordinates(self):
        return tuple([self.coord_x, self.coord_y])

    def get_orientation(self) -> str:
        return self.orientation

    def get_all_coordinates(self):
        """ Returns a list with all the coordinates that the piece
        covers. These are still the top left coordinates of each sub
        block within a piece. """
        if self.is_single:
            return [self.get_coordinates()]
        elif self.is_goal:
            return [self.get_coordinates(),
                    tuple([self.coord_x, self.coord_y + 1]),
                    tuple([self.coord_x + 1, self.coord_y]),
                    tuple([self.coord_x + 1, self.coord_y + 1])]
        elif self.orientation == 'h':
            return [self.get_coordinates(),
                    tuple([self.coord_x + 1, self.coord_y])]
        elif self.orientation == 'v':
            return [self.get_coordinates(),
                    tuple([self.coord_x, self.coord_y + 1])]
        else:
            pass

    def set_location(self, new_x, new_y):
        self.coord_x = new_x
        self.coord_y = new_y

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, \
                                       self.coord_x, self.coord_y,
                                       self.orientation)


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def show_grid(self):
        return self.grid

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()

    def display1(self):
        """
        Return the current board as a string.
        """
        board_str = ""
        for line in self.grid:
            board_str += "".join(line) + "\n"
        return board_str

    def find_empty(self):
        """Loops over the list in the grid, saves both indices in dictionary
        with the first empty space and second empty space with the keys
        'fs' and 'ss'"""
        dic = {}
        counter = 0
        for row, elem in enumerate(self.grid):
            for col, val in enumerate(elem):
                if val == '.':
                    if 'fs' not in dic:
                        dic['fs'] = [col, row]
                    else:
                        dic['ss'] = [col, row]
                        return dic
        return dic

    def is_empty(self, x, y):
        """Checks to see if the coordinates entered are an empty space"""
        try:
            if self.grid[y][x] == '.':
                return True
            else:
                return False
        except IndexError:
            return False

    def valid_move_right(self, piece: Piece):
        result = piece.get_all_coordinates()
        if piece.is_goal or piece.orientation == 'v':
            if self.is_empty(result[-1][0] + 1, result[-1][1]) \
                    and self.is_empty(result[-2][0] + 1, result[-2][1]) and \
                    not (result[-1][0] + 1 >= 4):
                return True
            else:
                return False
        elif piece.orientation == 'h' or piece.is_single:
            if self.is_empty(result[-1][0] + 1, result[-1][1]) and \
                    not (result[-1][0] + 1 >= 4):
                return True
            else:
                return False
        else:
            return False

    def move(self, piece: Piece, t: str):
        if t not in ['r', 'u', 'd', 'l']:
            return print('Not a valid expression for move to be made.')
        b = Board(copy.deepcopy(self.pieces))
        for p in b.pieces:
            if p.get_coordinates() == piece.get_coordinates():
                if t == 'r':
                    p.set_location(p.coord_x + 1, p.coord_y)
                    break
                elif t == 'l':
                    p.set_location(piece.coord_x - 1, piece.coord_y)
                elif t == 'u':
                    p.set_location(piece.coord_x, piece.coord_y - 1)
                else:  # has to be down
                    p.set_location(piece.coord_x, piece.coord_y + 1)

        b.__init__(b.pieces)
        return b

    def valid_move_left(self, piece: Piece):
        result = piece.get_all_coordinates()
        if not (piece.coord_x - 1 < 0):
            if (piece.is_single or piece.orientation == 'h') and \
                    self.is_empty(piece.coord_x - 1, piece.coord_y):
                # piece.set_location(piece.coord_x - 1, piece.coord_y)
                return True
            elif (piece.is_goal or piece.orientation == 'v') and \
                    self.is_empty(piece.coord_x - 1, piece.coord_y) \
                    and self.is_empty(result[1][0] - 1,
                                      result[1][1]):
                # piece.set_location(piece.coord_x - 1, piece.coord_y)
                return True
            else:
                return False

    def valid_move_up(self, piece: Piece):
        result = piece.get_all_coordinates()
        if not (piece.coord_y - 1 < 0):
            if (piece.is_single or piece.orientation == 'v') and \
                    self.is_empty(piece.coord_x, piece.coord_y - 1):
                return True
            elif piece.is_goal:
                if self.is_empty(piece.coord_x, piece.coord_y - 1) and \
                        self.is_empty(result[-2][0],
                                      result[-2][1] - 1):
                    return True

            elif piece.orientation == 'h':
                if self.is_empty(piece.coord_x, piece.coord_y - 1) and \
                        self.is_empty(result[-1][0],
                                      result[-1][1] - 1):
                    return True
            else:
                return False

    def valid_move_down(self, piece: Piece):
        result = piece.get_all_coordinates()
        if piece.is_goal:
            if not (result[-1][1] + 1 >= 5) and self.is_empty(result[-1][0],
                                                              result[-1][1] + 1) \
                    and self.is_empty(result[-3][0], result[-3][1] + 1):
                return True
            else:
                return False
        elif piece.orientation == 'h':
            if not (result[-1][1] + 1 >= 5) and self.is_empty(result[-1][0],
                                                              result[-1][1] + 1) \
                    and self.is_empty(result[-2][0], result[-2][1] + 1):
                return True
            else:
                return False
        elif piece.is_single or piece.orientation == 'v':
            if not (result[-1][1] + 1 >= 5) and \
                    self.is_empty(result[-1][0],
                                  result[-1][1] + 1):
                return True
            else:
                return False
        else:
            return False

    def goal_coord(self):
        for piece in self.pieces:
            if piece.is_goal:
                return piece.get_coordinates()

    def __eq__(self, other):
        if isinstance(other, Board):
            return self.grid == other.grid
        return False

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.grid))


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.

    def get_relevant_pieces(self):
        d = self.board.find_empty()
        lst = []
        val = d['fs']
        left = tuple([val[0] - 1, val[1]])
        right = tuple([val[0] + 1, val[1]])
        below = tuple([val[0], val[1] + 1])
        above = tuple([val[0], val[1] - 1])

        val1 = d['ss']
        left_2 = tuple([val1[0] - 1, val1[1]])
        right_2 = tuple([val1[0] + 1, val1[1]])
        below_2 = tuple([val1[0], val1[1] + 1])
        above_2 = tuple([val1[0], val1[1] - 1])

        for elem in self.board.pieces:
            if (left and left_2) in elem.get_all_coordinates():
                lst.append(elem)
            else:
                if left in elem.get_all_coordinates():
                    lst.append(elem)
                elif left_2 in elem.get_all_coordinates():
                    lst.append(elem)
                else:
                    pass

            if (right and right_2) in elem.get_all_coordinates():
                lst.append(elem)
            else:
                if right in elem.get_all_coordinates():
                    lst.append(elem)
                elif right_2 in elem.get_all_coordinates():
                    lst.append(elem)

            if (above and above_2) in elem.get_all_coordinates():
                lst.append(elem)
            else:
                if above in elem.get_all_coordinates():
                    lst.append(elem)
                elif above_2 in elem.get_all_coordinates():
                    lst.append(elem)

            if (below and below_2) in elem.get_all_coordinates():
                lst.append(elem)
            else:
                if below in elem.get_all_coordinates():
                    lst.append(elem)
                elif below_2 in elem.get_all_coordinates():
                    lst.append(elem)

        return lst

    def __eq__(self, other):
        return self.board.display1() == other.board.display1()

    def generate_successors(self):
        l = self.get_relevant_pieces()
        d = self.board
        final = []

        for p in l:
            if d.valid_move_right(p):
                b = d.move(p, 'r')
                s = State(b, self.f + 1, self.depth + 1, d)
                if s not in final:
                    final.append(s)
            if d.valid_move_left(p):
                b = d.move(p, 'l')
                s = State(b, self.f + 1, self.depth + 1, d)
                if s not in final:
                    final.append(s)
            if d.valid_move_up(p):
                b = d.move(p, 'u')
                s = State(b, self.f + 1, self.depth + 1, d)
                if s not in final:
                    final.append(s)
            if d.valid_move_down(p):
                b = d.move(p, 'd')
                s = State(b, self.f + 1, self.depth + 1, d)
                if s not in final:
                    final.append(s)

        return final

    def is_goal(self):
        for p in self.board.pieces:
            if p.is_goal and p.coord_x == 1 and p.coord_y == 3:
                return True
        return False

    def __lt__(self, other):
        return self.f < other.f


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)

    return board


def dfs(init_state: State):
    frontier = [init_state]
    explored = dict()
    solution = []
    s_key = None
    explored[init_state.board.display1()] = None

    while bool(frontier):
        curr = frontier.pop()
        if curr.is_goal():
            solution.append(curr)
            s_key = curr.board.display1()
            break
        for successor in curr.generate_successors():
            try:
                if successor.board.display1() not in explored:
                    explored[successor.board.display1()] = curr.board.display1()
                    frontier.append(successor)
            except TypeError:
                break

    while s_key is not None:
        solution.append(s_key)
        s_key = explored[s_key]

    solution.reverse()
    solution.pop()
    return solution


def output(filename: str, solution: List):
    f = open(filename, "w")
    for elem in solution:
        if type(elem) is str:
            f.write(elem)
        if type(elem) is State:
            f.write(str(elem))
        f.write("\n")
    f.close()


def manhattan(state: State):
    coordinates = state.board.goal_coord()
    return abs(1 - coordinates[0]) + abs(3 - coordinates[1])


def h_search(init_state: State, heuristic_func):
    frontier = []
    explored = dict()
    heappush(frontier, (0, init_state))
    explored[init_state.board.display1()] = None
    solution = []
    s_key = None

    while bool(frontier):
        curr = heappop(frontier)
        if curr[1].is_goal():
            s_key = curr[1].board.display1()
            break

        for successor in curr[1].generate_successors():
            curr_cost = cost(curr[1], explored) + 1
            if successor.board.display1() not in explored:
                explored[successor.board.display1()] = curr[1].board.display1()
                priority = curr_cost + heuristic_func(successor)
                heappush(frontier, (priority, successor))

    while s_key is not None:
        solution.append(s_key)
        s_key = explored[s_key]

    solution.reverse()
    return solution


def cost(state: State, explored: dict):
    parent = explored[state.board.display1()]
    a = 0
    while parent is not None:
        a += 1
        parent = explored[parent]

    return a


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)
    state = State(board, 0, 0)

    if args.algo == 'astar':
        result = h_search(state, manhattan)
        print(len(result))
        output(args.outputfile, result)
    if args.algo == 'dfs':
        result = dfs(state)
        output(args.outputfile, result)
