'''
Class that represent the features and methods for a Human Board game player
 Programmed by Bjørnar Tessem, Sept-Oct 2022
'''

from BoardPlayer import BoardPlayer
from Board import loc_print


class BoardHumanPlayer(BoardPlayer):
    '''
        Implementation of human player
    '''

    def __init__(self, the_mark):
        '''
        constructor
        :param the_mark (i.e.) either CROSS or RING:
        '''
        super(BoardHumanPlayer, self).__init__(the_mark)
        self.name = "You"

    def move(self, board, moves):
        '''
        Selects a possible move for this player and modifies the game status
        :param board: current board of game
        :param moves: the possible moves for the player in this position
        :return: the selected move
        '''

        # print all possible moves
        print("Possible moves:")

        # print all moves with an index for selection
        for i in range(len(moves)):
            print(
                str(i) + ': ' + loc_print(moves[i].last_move_origin) + ' to ' + loc_print(moves[i].last_move_endpoint))
        accepted = False
        index = -1

        # check move and accept (very simple input checks)
        while not accepted:
            answer = input("Choose move (integer) > ")
            try:
                # accept only integer input
                index = int(answer)

            except:
                print(answer, " is not a number")
                continue

            if -1 < index < len(moves):
                # accept only indices to a real move
                accepted = True
            else:
                print(answer, " illegal move!")

        # return the chosen move
        return moves[index]
