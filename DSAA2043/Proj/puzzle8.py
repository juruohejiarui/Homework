import heapq
import copy
import argparse
import time

# trg = [(1, 1), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
trg = [(2, 2), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
mvVec = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class State :
    def _calcHsPred(self) :
        self.hs = self.pred = 0
        for i in range(3) :
            for j in range(3) :
                self.hs = self.hs * 10 + self.mtx[i][j]
                trgX, trgY = trg[self.mtx[i][j]]
                self.pred += abs(i - trgX) + abs(j - trgY)
    def __init__(self, mtx) :
        if isinstance(mtx, list) :
            self.mtx = mtx.copy()        
            self._calcHsPred()
            
    def __getitem__(self, x) -> int :
        return self.mtx[x[0]][x[1]]
    def __setitem__(self, x, y) :
        self.mtx[x[0]][x[1]] = y
    def __lt__(self, other) -> bool :
        return self.hs < other.hs
    def __eq__(self, other) -> bool :
        if not isinstance(other, State) : return False
        else : 
            return self.hs == other.hs
    
    def find0(self) :
        for i in range(3) :
            for j in range(3) :
                if self.mtx[i][j] == 0 : return (i, j)
        return (-1, -1)
    def swap(self, p1 : tuple[int, int], p2 : tuple[int, int]) :
        self.mtx[p1[0]][p1[1]], self.mtx[p2[0]][p2[1]] = self.mtx[p2[0]][p2[1]], self.mtx[p1[0]][p1[1]]
        self._calcHsPred()
    def print(self, outputFile = None) :
        if outputFile == None :
            for i in range(3) :
                for j in range(3) :
                    print(f"{self.mtx[i][j] if self.mtx[i][j] != 9 else 0} ", end='')
                print('')
        else :
            for i in range(3) :
                for j in range(3) :
                    outputFile.write(f"{self.mtx[i][j] if self.mtx[i][j] != 9 else 0} ")
                outputFile.write('\n')
    def isFinish(self) :
        return self.pred == 0
            
def astar(initState : list, outputPath : str) :
    vis = {}
    initState : State = State(initState)
    que : list[tuple[int, State]] = [(initState.pred, initState)]
    preState = {}
    vis[initState.hs] = 0
    finishState : State = None
    finDis = -1
    while len(que) :
        (dis, u) = heapq.heappop(que)
        if u.isFinish() :
            finishState = u
            finDis = dis
            break
        x, y = u.find0()
        dis -= u.pred
        for dx, dy in mvVec :
            x1, y1 = x + dx, y + dy
            if x1 < 0 or y1 < 0 or x1 > 2 or y1 > 2 :
                continue
            v = copy.deepcopy(u)
            v.swap((x, y), (x1, y1))
            hsV = v.hs
            if hsV not in vis or vis[hsV] > dis + 1:
                vis[hsV] = dis + 1
                preState[hsV] = u
                heapq.heappush(que, (v.pred + dis + 1, v))
    if outputPath == None :
        if finDis == -1 :
            print("-1")
        else :
            print(finDis)
    else :
        outputFile = open(outputPath, 'w')
        # print solution
        if finDis == -1 :
            outputFile.write("UNSOLVABLE\n")
        else :
            initState.print(outputFile)
            # print(finDis)
            ansStk : list[State] = []
            state : State = finishState
            while state != initState :
                ansStk.append(state)
                state = preState[state.hs]
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
    # # st = time.time()
    # astar(ans, None)

    
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
    
    # print(time.time() - st)
