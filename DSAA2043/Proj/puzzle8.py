import heapq

class State :
    def __init__(self, mtx : list) :
        self.mtx = mtx.copy()
    def dis(self) :
        ans = 0
        for i in range(3) :
            for j in range(3) :
                trgX, trgY = (self.mtx[i][j] - 1) // 3, (self.mtx[i][j] - 1) % 3
                ans += abs(i - trgX) + abs(j - trgY)
        return ans
    def __getitem__(self, x) :
        return self.mtx[x[0]][x[1]]
    def __setitem__(self, x, y) :
        self.mtx[x[0]][x[1]] = y

state = State([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
state[1, 2] = 114
print(state.mtx)