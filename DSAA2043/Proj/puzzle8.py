import heapq

class State :
    def __init__(self, mtx : list) :
        self.mtx = mtx.copy()
    def dis(self) -> int :
        ans = 0
        for i in range(3) :
            for j in range(3) :
                trgX, trgY = (self.mtx[i][j] - 1) // 3, (self.mtx[i][j] - 1) % 3
                ans += abs(i - trgX) + abs(j - trgY)
        return ans
    def __getitem__(self, x) -> int :
        return self.mtx[x[0]][x[1]]
    def __setitem__(self, x, y) :
        self.mtx[x[0]][x[1]] = y
    def find9(self) :
        for i in range(3) :
            for j in range(3) :
                if self.mtx[i][j] == 9 : return (i, j)
        return None
    def swap(self, p1 : tuple[int, int], p2 : tuple[int, int]) :
        self.mtx[p1[0]][p1[1]], self.mtx[p2[0]][p2[1]] = self.mtx[p2[0]][p2[1]], self.mtx[p1[0]][p1[1]]
    def toInt(self) :
        val = 0
        for i in range(3) :
            for j in range(3) :
                val = val * 10 + self.mtx[i][j]
        return val
    def print(self) :
        for i in range(3) :
            for j in range(3) :
                print(self.mtx[i][j], end=' ')
            print('')
            
def astar(initState : list) -> list[State] :
    vis = set()
    que = [State(initState)]
    