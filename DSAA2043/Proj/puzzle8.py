import heapq
import copy
import argparse

trg = [(2, 2), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

class State :
    def __init__(self, mtx : list) :
        self.mtx = mtx.copy()
    def dis(self) -> int :
        global trg
        ans = 0
        for i in range(3) :
            for j in range(3) :
                trgX, trgY = trg[self.mtx[i][j]]
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
    
    def find0(self) :
        for i in range(3) :
            for j in range(3) :
                if self.mtx[i][j] == 0 : return (i, j)
        return (-1, -1)
    def swap(self, p1 : tuple[int, int], p2 : tuple[int, int]) :
        self.mtx[p1[0]][p1[1]], self.mtx[p2[0]][p2[1]] = self.mtx[p2[0]][p2[1]], self.mtx[p1[0]][p1[1]]
    def toInt(self) :
        val = 0
        for i in range(3) :
            for j in range(3) :
                val = val * 10 + self.mtx[i][j]
        return val
    def print(self, outputFile) :
        for i in range(3) :
            for j in range(3) :
                outputFile.write(f"{self.mtx[i][j] if self.mtx[i][j] != 9 else 0} ")
            outputFile.write('\n')
    def isFinish(self) :
        return self.dis() == 0
            
def astar(initState : list, outputPath : str) :
    vis = {}
    initState : State = State(initState)
    que : list[tuple[int, State]] = [(initState.dis(), initState)]
    preState = {}
    vis[initState.toInt()] = 0
    finishState : State = None
    finDis = -1
    while len(que) :
        (dis, u) = heapq.heappop(que)
        if u.isFinish() :
            finishState = u
            finDis = dis
            break
        x, y = u.find0()
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
                if hsV not in vis or vis[hsV] > dis + 1:
                    vis[hsV] = dis + 1
                    preState[hsV] = u
                    heapq.heappush(que, (v.dis() + dis + 1, v))
    outputFile = open(outputPath, "w")
    # print solution
    if finDis == -1 :
        outputFile.write("UNSOLVABLE\n")
    else :
        initState.print(outputFile)
        # print(finDis)
        ansStk : list[State] = []
        state : State = finishState
        while True :
            ansStk.append(state)
            state = preState[state.toInt()]
            if state == initState : break
        while len(ansStk) > 0 :
            outputFile.write("\n")
            ansStk.pop().print(outputFile)


if __name__ == "__main__" :
    # seq = int((input()[::-1]))
    # ans = [None] * 3
    # for i in range(3) :
    #     ans[i] = [0, 0, 0]
    #     for j in range(3) :
    #         ans[i][j] = seq % 10 
    #         seq //= 10
    # # print(ans)
    # astar(ans)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    arg = parser.parse_args()
    
    initState = [None] * 3
    inputFile = arg.input_file
    outputFile = arg.output_file
    inputFile = open(inputFile, "r")
    for i in range(3) :
        initState[i] = list(map(int, inputFile.readline().split()))
    astar(initState, outputFile)
        