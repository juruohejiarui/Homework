# Detailed

## Implementation
First, I implemented a structure called $\texttt{State}$ , which helps me to manage the value of 
"prediction" ($\texttt{pred}$), hash value of the state ($\texttt{hs}$) and the operation ($\texttt{swap}()$) and so on.

The calculation of prediction is Manhattan priority function, which is mentioned in document of requirement.

Then comes to the A* algorithm. I use a heap, whose elements are tuples. For each tuple, the first entry is the sum of prediction and distance from the initial state. And the second entry is a $\texttt{State}$ structure. Additionally, a hash table $\texttt{vis}$ is applied to store the minimum of steps from the initial state to a specific state. For each state, the program will try 4 operation (Up, Down, Left, Right) and check whether it can get to a new state or can get to a visited state with less steps, if so, then add a new tuple to the heap. It seems that we can just check whether we have visited the state, which is also correct, but I prefer to the previous mechanism.

## Correctness
A* can reach to any valid state from initial one.

Let $h(n)$ be the function of prediction (Manhattan priority function) . Then we have 
$$
h(n)\leq \text{actual cost from }n\text{ to the goal}
$$
This can guarantee that we can always find the optimal solution.
Additionally, we have
$$
h(n)\leq \mathrm{dis}(n, m) + h(m)
$$
where $m$ is reachable from $n$.
This statement guarantees that the prediction of cost $\mathrm{dis}(\text{initial state},n)+h(n)$ on a path is not decreasing, and leads to that the program will not visit a node again when a better path is found.

## Complexity

Let $n$ be the valid state from the intial state. For the worst cases, when we have to visit all the valid states. then the space complexity is $O(n)$ . Since each state is visited for constant times, and for each state, some adjacent states are pushed into heap, then the time complexity is $O(n\log_2 n)$

For most of the actual cases, we can just visit the states that not far from initial state, which is makes the complexity much smaller than the worst cases.
# Execution

```bash
$ python3 ./puzzle8.py "input_file" "output_file"
```