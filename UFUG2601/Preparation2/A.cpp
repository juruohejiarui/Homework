#include <bits/stdc++.h>

using namespace std;

bool OutOfRange(int x, int y) { return false; }

void Search(const vector< vector<int> > &arr, int x, int y) {
    if (OutOfRange(x, y)) return ;
    Search(arr, x + 1, y);
    Search(arr, x, y + 1);
    Search(arr, x - 1, y);
    Search(arr, x, y - 1);
}
int main() {
    return 0;
}