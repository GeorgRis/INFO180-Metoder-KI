'''
A class that defines important features and methods for all player objects
Programmed by Bjørnar Tessem, Sept-Oct 2022
'''

class BoardPlayer():
    '''
    Abstract class that represents a player, either human or machine
    '''

    def __init__(self, the_mark):
        '''
        constructor sets a mark for the player and a name
        '''
        self.mark = the_mark
        self.name = ""

    def move(self, board, moves):
        '''
        Selects a possible move for this player and modifies the game status
        :param game_status: current status of game
        :param i: candidate list 1 or 2
        :return: nothing
        '''
        pass