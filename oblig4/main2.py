'''
 A main file for running the board game with human player
 Programmed by Bjørnar Tessem, Sept-Oct 2022
'''

from PlayBoardGame import PlayBoardGame
from Board import Board, NOT_TERMINAL, CROSS, RING
from BoardGameTreeNode import BoardGameTreeNode

simple = 0
advanced = 0

def spill():
    global advanced
    global simple

    game = PlayBoardGame()
    # as long as not finished
    while not (game.finished()):
        game.print_status()

        # brint game board and other information


        # allow the player in turn to move, either human or computer
        game.select_move()
    res = game.print_result()
    if res == game.player1.name:
        advanced += 1
    elif res == game.player2.name:
        simple += 1


def noe():
    global advanced
    global simple
    for size in range(5,8):
        for dybde in range(5,8):
            Board.GAMESIZE = size
            BoardGameTreeNode.MAX_DEPTH = dybde
            spill()

game = PlayBoardGame()

if __name__ == "__main__":
    noe()
    print(game.player2.name, "-", advanced, game.player1.name, "-", simple)



