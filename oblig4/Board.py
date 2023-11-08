'''
    class Board - class representing the board of a small board game
    Programmed by Bjørnar Tessem - Sept 2022
'''
from copy import deepcopy

# Numbers for representing the possible values of squares on the board, as well as cross or ring wins
CROSS = 'X'
RING = 'O'
EMPTY = '_'

# Numbers for representing the various states and outcomes of the game
NOT_TERMINAL = 'N'
DRAW = '='

# Number for the size of the square board
GAMESIZE = 5

# Writes a code for a square on the board in the form capital letter (column) and number (row)
# The board index is given as a pair (row, column)
def loc_print(board_index):
    # column in board_index[1] and row in board_index[0]
    return chr(ord('A') + board_index[1]) + str(board_index[0] + 1)

class Board:
    '''
    A class that handles the board, its status and possible moves
    '''
    def __init__(self, initial_board = None):
        '''
        Creates a new board
        :param initial_board: a board sent for copying
        '''
        if initial_board is None:
            # makes a start board
            self.the_grid = self.make_initial_grid()
            self.to_play = CROSS

            # it is convenient to remember the last move
            self.last_move_origin = (-1,-1)
            self.last_move_endpoint = (-1,-1)

            # the game is (of course) not finished initially
            self.game_status = NOT_TERMINAL
        else:
            # copies the board from the argument
            self.the_grid = deepcopy(initial_board.the_grid)
            self.to_play = initial_board.to_play
            self.last_move_origin = initial_board.last_move_origin
            self.last_move_origin = initial_board.last_move_endpoint
            self.game_status = initial_board.game_status

    def finished(self):
        '''
        checks if board is a finished game
        :return: true if one of DRAW, CROSS or RING
        '''
        return self.game_status != NOT_TERMINAL

    def make_initial_grid(self):
        '''
        construct an initial board of the game
        :return: an initial board
        '''
        # create the empty board
        board_grid = []
        for i in range(GAMESIZE):
            row = []
            for j in range(GAMESIZE):
                row.append(EMPTY)
            board_grid.append(row)

        # fill the first and last row
        for i in range(GAMESIZE):
            board_grid[0][i] = CROSS
            board_grid[GAMESIZE-1][i] = RING
        return board_grid

    def check_and_generate_moves(self):
        '''
        generates the moves and if no moves suggests that the game is a draw or a win
        :return: all legal moves in the state
        '''
        game_result = NOT_TERMINAL

        for j in range(GAMESIZE):
            # checks if any of players have reached the opposite row
            if self.the_grid[GAMESIZE-1][j] == CROSS:
                game_result = CROSS
            if self.the_grid[0][j] == RING:
                game_result = RING

        # if not a winning position for any part generate moves
        if game_result == NOT_TERMINAL:

            # generate cross moves
            if self.to_play == CROSS:
                legal_children = self.generate_cross_moves()

            # generate ring moves
            else:
                legal_children = self.generate_ring_moves()

        # no legal children if game is won
        else:
            legal_children = []

        # if there are no moves the game is a draw
        if len(legal_children) == 0 and game_result == NOT_TERMINAL:
            game_result = DRAW

        # set the game result for the state
        self.game_status = game_result

        # return the legal children
        return legal_children

    def generate_cross_moves(self):
        '''
        generates moves if cross is to move
        :return: a collection of legal moves for cross
        '''
        moves = self.cross_generate_forward_moves() + \
                self.cross_generate_left_moves() + \
                self.cross_generate_right_moves()
        return moves

    def generate_ring_moves(self):
        '''
        generates moves if ring is to move
        :return: a collection of legal moves for ring
        '''
        moves = self.ring_generate_forward_moves() + \
                self.ring_generate_left_moves() + \
                self.ring_generate_right_moves()
        return moves


    def __str__(self):
        '''
        a print representation of the board
        :return: a string that prints the board
        '''
        return_str = "\nTurn: "

        # whose turn is it
        if self.to_play == CROSS:
            return_str = return_str + 'X\n'
        else:
            return_str = return_str + 'O\n'

        # print the board grid
        for i in range(GAMESIZE-1,-1,-1):
            # row number first
            return_str = return_str + str(i + 1) + '  '
            for j in range(GAMESIZE):
                if self.the_grid[i][j] == EMPTY:
                    return_str = return_str + '- '
                elif self.the_grid[i][j] == CROSS:
                    return_str = return_str + 'X '
                else:
                    return_str = return_str + 'O '
            return_str = return_str + '\n'
        return_str = return_str + '  '

        # column letters
        for i in range(GAMESIZE):
            return_str = return_str + ' ' + chr(ord('A')+i)
        return_str = return_str + '\n'
        return return_str

    def cross_generate_forward_moves(self):
        '''
        create moves for cross straight ahead
        :return: a list of moves
        '''
        moves = []
        for i in range(GAMESIZE-1):
            for j in range(GAMESIZE):
                if self.the_grid[i][j] == CROSS:
                    if self.the_grid[i + 1][j] == EMPTY:
                        new_board = Board(self)
                        new_board.the_grid[i][j] = EMPTY
                        new_board.the_grid[i+1][j] = CROSS
                        new_board.to_play = RING
                        new_board.last_move_origin = (i,j)
                        new_board.last_move_endpoint = (i+1,j)
                        moves.append(new_board)
        return moves

    def cross_generate_left_moves(self):
        '''
        create moves for cross diagonally left
        :return: a list of moves
        '''
        moves = []
        for i in range(GAMESIZE-2):
            for j in range(2,GAMESIZE):
                if self.the_grid[i][j] == CROSS:
                    if self.the_grid[i + 2][j-2] == EMPTY and self.the_grid[i + 1][j - 1] != EMPTY:
                        new_board = Board(self)
                        new_board.the_grid[i][j] = EMPTY
                        new_board.the_grid[i + 2][j-2] = CROSS
                        if new_board.the_grid[i + 1][j-1] == RING:
                            new_board.the_grid[i + 1][j - 1] = EMPTY
                        new_board.to_play = RING
                        new_board.last_move_origin = (i,j)
                        new_board.last_move_endpoint = (i+2,j-2)
                        moves.append(new_board)
        return moves

    def cross_generate_right_moves(self):
        '''
        create moves for cross diagonally right
        :return: a list of moves
        '''
        moves = []
        for i in range(GAMESIZE-2):
            for j in range(GAMESIZE-2):
                if self.the_grid[i][j] == CROSS:
                    if self.the_grid[i + 2][j+2] == EMPTY and self.the_grid[i + 1][j + 1] != EMPTY:
                        new_board = Board(self)
                        new_board.the_grid[i][j] = EMPTY
                        new_board.the_grid[i + 2][j+2] = CROSS
                        if new_board.the_grid[i + 1][j+1] == RING:
                            new_board.the_grid[i + 1][j + 1] = EMPTY
                        new_board.to_play = RING
                        new_board.last_move_origin = (i,j)
                        new_board.last_move_endpoint = (i+2,j+2)
                        moves.append(new_board)
        return moves

    def ring_generate_forward_moves(self):
        '''
        create moves for ring straight ahead
        :return: a list of moves
        '''
        moves = []
        for i in range(1,GAMESIZE):
            for j in range(GAMESIZE):
                if self.the_grid[i][j] == RING:
                    if self.the_grid[i -1 ][j] == EMPTY:
                        new_board = Board(self)
                        new_board.the_grid[i][j] = EMPTY
                        new_board.the_grid[i - 1 ][j] = RING
                        new_board.to_play = CROSS
                        new_board.last_move_origin = (i,j)
                        new_board.last_move_endpoint = (i-1,j)
                        moves.append(new_board)
        return moves

    def ring_generate_left_moves(self):
        '''
        create moves for ring diagonally left
        :return: a list of moves
        '''
        moves = []
        for i in range(2,GAMESIZE):
            for j in range(2, GAMESIZE):
                if self.the_grid[i][j] == RING:
                    if self.the_grid[i - 2][j - 2] == EMPTY and self.the_grid[i - 1][j - 1] != EMPTY:
                        new_board = Board(self)
                        new_board.the_grid[i][j] = EMPTY
                        new_board.the_grid[i - 2][j - 2] = RING
                        if new_board.the_grid[i - 1][j - 1] == CROSS:
                            new_board.the_grid[i - 1][j - 1] = EMPTY
                        new_board.to_play = CROSS
                        new_board.last_move_origin = (i,j)
                        new_board.last_move_endpoint = (i-2,j-2)
                        moves.append(new_board)
        return moves

    def ring_generate_right_moves(self):
        '''
        create moves for ring diagonally right
        :return: a list of moves
        '''
        moves = []
        for i in range(2,GAMESIZE):
            for j in range(GAMESIZE-2):
                if self.the_grid[i][j] == RING:
                    if self.the_grid[i - 2][j + 2] == EMPTY and self.the_grid[i - 1][j + 1] != EMPTY:
                        new_board = Board(self)
                        new_board.the_grid[i][j] = EMPTY
                        new_board.the_grid[i - 2][j + 2] = RING
                        if new_board.the_grid[i - 1][j + 1] == CROSS:
                            new_board.the_grid[i - 1][j + 1] = EMPTY
                        new_board.to_play = CROSS
                        new_board.last_move_origin = (i,j)
                        new_board.last_move_endpoint = (i-2,j+2)
                        moves.append(new_board)
        return moves
