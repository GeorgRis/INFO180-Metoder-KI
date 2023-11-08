'''
 A main file for running the board game with human player
 Programmed by Bjørnar Tessem, Sept-Oct 2022
'''

from PlayBoardGame import PlayBoardGame

# create a game playing object
game = PlayBoardGame()

# allow person to select starter
game.select_starter()

# as long as not finished
while not(game.finished()):
    # brint game board and other information
    game.print_status()

    # allow the player in turn to move, either human or computer
    game.select_move()

game.print_result()