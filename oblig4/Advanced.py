
from Board import CROSS, RING,GAMESIZE
from BoardComputerPlayer import BoardComputerPlayer


class Advanced(BoardComputerPlayer):

    def __init__(self, the_mark):
        '''
        Constructor
        :param compatibility_score_set:
        '''
        super(Advanced, self).__init__(the_mark)
        self.name = "Advanced"

    def evaluate_game_status(self, a_board):
        max_cross_row = 0
        max_ring_row = GAMESIZE-1
        cross_count = 0
        ring_count = 0
        score = 0
        for i in range(GAMESIZE):
            for j in range(GAMESIZE):
                if a_board.the_grid[i][j] == CROSS:
                    cross_count += 1
                    if i > max_cross_row:
                        if 1 < j < GAMESIZE - 1:
                            max_cross_row = i
                if a_board.the_grid[i][j] == RING:
                    ring_count += 1
                    if i < max_ring_row:
                        if 1 < j < GAMESIZE-1:
                            max_ring_row = i

        if self.mark == CROSS:
            score += max_cross_row - (GAMESIZE-1-max_ring_row)
            score += cross_count - ring_count

        if self.mark == RING:
            score = (GAMESIZE-1-max_ring_row) - max_cross_row
            score += ring_count - cross_count
        return score
