from Oblig1_180.heuristic import Heuristic
from Oblig1_180.make_grid import SIZE
from math import sqrt


SIZE = SIZE - 1
class ZeroH(Heuristic):

    @staticmethod
    def h(node):
        return 0


class notZeroH(Heuristic):

    @staticmethod
    def h(node):
        avstand_i = SIZE-node.i
        avstand_j = SIZE-node.j

        if (avstand_j) > (avstand_i):
            return avstand_j
        else: return avstand_i

class EucZeroH(Heuristic):

    @staticmethod
    def h(node):
        avstand_i = SIZE-node.i
        avstand_j = SIZE-node.j

        euc = sqrt((avstand_i**2)+(avstand_j**2))
        return euc

class notX3ZeroH(Heuristic):

    @staticmethod
    def h(node):
        avstand_i = (SIZE-node.i)*3
        avstand_j = (SIZE-node.j)*3

        if (avstand_j) > (avstand_i):
            return avstand_j
        else: return avstand_i