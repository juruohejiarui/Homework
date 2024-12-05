import heapq
import copy

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
    def __lt__(self, other) -> bool :
        return self.toInt() < other.toInt()
    def __eq__(self, other) -> bool :
        if not isinstance(other, State) : return False
        else : 
            return self.toInt() == other.toInt()
    
    def find9(self) :
        for i in range(3) :
            for j in range(3) :
                if self.mtx[i][j] == 9 : return (i, j)
        return (-1, -1)
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
                print(self.mtx[i][j] if self.mtx[i][j] != 9 else 0, end=' ')
            print('')
    def isFinish(self) :
        return self.toInt() == 123456789
            
def astar(initState : list) :
    vis = set()
    initState : State = State(initState)
    que : list[tuple[int, State]] = [(initState.dis(), initState)]
    preState = {}
    vis.add(initState.toInt())
    finishState = None
    while len(que) :
        (dis, u) = heapq.heappop(que)
        if u.isFinish() :
            finishState = u
            break
        x, y = u.find9()
        dis -= u.dis()
        # print(f"dis={dis}")
        # u.print()
        for dx in range(-1, 2) :
            for dy in range(-1, 2) :
                if abs(dx) + abs(dy) != 1 : continue
                x1, y1 = x + dx, y + dy
                if x1 < 0 or y1 < 0 or x1 > 2 or y1 > 2 :
                    continue
                v = copy.deepcopy(u)
                v.swap((x, y), (x1, y1))
                hsV = v.toInt()
                if hsV not in vis :
                    vis.add(hsV)
                    preState[hsV] = u
                    heapq.heappush(que, (v.dis() + dis + 1, v))
    # print solution
    if finishState == None :
        print("<No Solution>")
    else :
        ansStk : list[State] = []
        state : State = finishState
        while True :
            ansStk.append(state)
            state = preState[state.toInt()]
            if state == initState : break
        while len(ansStk) > 0 :
            ansStk.pop().print()
            print('')


if __name__ == "__main__" :
    ans = [None] * 3
    for i in range(3) :
        ans[i] = list(map(int, input().split()))
    for i in range(3) :
        for j in range(3) :
            if ans[i][j] == 0 : ans[i][j] = 9
    astar(ans)
        