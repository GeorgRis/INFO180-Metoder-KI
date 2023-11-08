'''
Class that represent the features and methods for a Simple Computer Board game player
 Programmed by Bjørnar Tessem, Sept-Oct 2022
'''


from Board import CROSS, RING, GAMESIZE
import Board
from BoardComputerPlayer import BoardComputerPlayer


class SimpleBoardPlayer(BoardComputerPlayer):

    def __init__(self, the_mark):
        '''
        Constructor
        :param compatibility_score_set:
        '''
        super(SimpleBoardPlayer, self).__init__(the_mark)
        self.name = "Simple"

    def evaluate_game_status(self, a_board):
        max_cross_row = 0
        max_ring_row = GAMESIZE-1
        for i in range(GAMESIZE):
            for j in range(GAMESIZE):
                if a_board.the_grid[i][j] == CROSS:
                    if i > max_cross_row:
                        max_cross_row = i
                if a_board.the_grid[i][j] == RING:
                    if i < max_ring_row:
                        max_ring_row = i
        score = 0
        if self.mark == CROSS:
            score = max_cross_row - (GAMESIZE-1-max_ring_row)
        if self.mark == RING:
            score = (GAMESIZE-1-max_ring_row) - max_cross_row
        return score


