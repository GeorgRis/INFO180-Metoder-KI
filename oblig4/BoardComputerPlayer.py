'''
Class that represent the features and methods for a Computer Board game player
 Programmed by Bjørnar Tessem, Sept-Oct 2022
'''

from BoardGameTreeNode import BoardGameTreeNode
from BoardPlayer import BoardPlayer

from Board import loc_print


class BoardComputerPlayer(BoardPlayer):
    '''
    An abstract computer player class
    '''

    def __init__(self, the_mark):
        '''
        Constructor
        :param the_mark: CROSS or RING
        '''
        super(BoardComputerPlayer, self).__init__(the_mark)
        self.name = "Comp"


    def move(self, the_board, the_moves):
        '''
           Use an algorithm to find a good candidate move
            :param the_board: the current board
            :param the_moves: the possible moves
            :return: a selected move
        '''

        # select one of the moves
        selected_candidate_idx = self.select_candidate(the_board)

        print(self.mark + ' chose ' + loc_print(the_moves[selected_candidate_idx].last_move_origin) + ' to ' + loc_print(the_moves[selected_candidate_idx].last_move_endpoint))
        return the_moves[selected_candidate_idx]

    def select_candidate(self, a_board):
        '''
        Runs alfa beta search to find candidate
        :param a_board: find best move for this board from a search
        :return: the best move
        '''
        root = BoardGameTreeNode(a_board)
        root.computer_player = self

        root.search()

        return root.best_move

    def evaluate_game_status(self):
        '''
        method to be implemented in each subclass
        '''
        pass

