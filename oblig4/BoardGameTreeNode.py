'''
Representation of a game tree with alpha beta search function
 Programmed by Bjørnar Tessem, Sept-Oct 2022
'''

from Board import DRAW, NOT_TERMINAL


class BoardGameTreeNode:
    '''
    A class that represents a node and also runs alfabet for that node
    '''

    # Max depth of search for the computer players
    MAX_DEPTH = 5


    def __init__(self, a_board):
        '''
        Constructor
        :param a_board: the board for the node
        '''
        self.MAX = False # not MAX by default
        self.the_board = a_board
        self.depth = 0 # depth = 0 by default
        self.best_move = -1 # no best moves
        self.score = 0 # score of game is DRAW
        self.computer_player = None # computer_player need to be assigned
        self.possible_moves = [] # no possible moves

    def leaf(self):
        '''
        checks if a node is a leaf in the game tree, either by being finished or being at MAX_DEPTH
        :return:
        '''

        # find possible moves, if none the board will have a status indicating that we are finished
        self.possible_moves = self.the_board.check_and_generate_moves()

        # if game is finished  we are at a leaf
        if self.the_board.finished():
            return True

        # if depth is MAX we are at a leaf
        if self.depth >= BoardGameTreeNode.MAX_DEPTH:
            return True

        # otherwise not
        return False

    def score_leaf(self):
        '''
        Evaluates the node
        :return:the value of the game as seen from MAX
        '''
        if self.the_board.finished():
            return self.game_score(self.the_board.game_status)
        else:
            return self.evaluate_game_status()

    def evaluate_game_status(self):
        '''
        Evaluation of game status depends on who is the player  - max or min
        :return: MAX's score of the position
        '''
        val = self.computer_player.evaluate_game_status(self.the_board)
        if self.MAX:
            return val
        else:
            return -1 * val

    def search(self):
        '''
        Initiate an alpha beta search
        :return:
        '''
        self.MAX = True
        self.alpha_beta(self, -99999999, 999999999)

    def alpha_beta(self, node, alpha, beta):
        '''
        The actual alpha beta algorithm
        :param node: the node in the search tree
        :param alpha: the lower boundary for MAX's game score
        :param beta: the upper boundary for MAX's game score
        :return: a value for MAX's game score
        '''

        # score the leaf and return
        if node.leaf():
            return node.score_leaf()

        # if not generate the children from the moves the node has
        children = node.generate_children()

        if node.MAX: # if MAX node
            for i in range(len(children)):
                # run through all children
                child = children[i]
                score = self.alpha_beta(child, alpha, beta)
                if score > alpha:
                    # then we have a better move for MAX
                    alpha = score
                    node.best_move = i
                if alpha >= beta:
                    # this move is unrealistic as MIN would not allow to end up here
                    # prune search
                    return beta
            node.score = alpha
            return alpha
        else:
            for i in range(len(children)):
                # run through all children
                child = children[i]
                score = self.alpha_beta(child, alpha, beta)
                if score < beta:
                    # we have a better move for MIN
                    beta = score
                    node.best_move = i
                if alpha >= beta:
                    # this move is unrealistic as MAX would not allow to end up here
                    # prune search
                    return alpha
            node.score = beta
            return beta

    def generate_children(self):
        '''
        generates the children nodes of a node in a search tree
        from the node's possible moves
        The list ov moves has already been created at this point and is found in
        self.possible_moves
        :return:
        '''
        children = []
        for a_board in self.possible_moves:
            # for each move make a new node
            # and initialise its values
            new_node = BoardGameTreeNode(a_board)
            new_node.computer_player = self.computer_player
            new_node.depth = self.depth + 1
            if self.MAX:
                new_node.MAX = False
            else:
                new_node.MAX = True
            children.append(new_node)
        return children

    def game_score(self, game_status):
        '''
        givens a numerical score for the game
        :param game_status: the final result of the game
        :return:
        '''
        if game_status == DRAW or game_status == NOT_TERMINAL:
            return 0
        if game_status == self.computer_player.mark:
            # subtracts depth just to ensure that the players go for the fastest wins
            return 10000-self.depth
        else:
            return -10000+self.depth
