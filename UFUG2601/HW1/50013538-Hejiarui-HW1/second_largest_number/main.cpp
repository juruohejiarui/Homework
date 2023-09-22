#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>

using namespace std;

const int INF = 1e8;
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int mx = -INF, smx = -INF;
    int n; scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
        int x; scanf("%d", &x);
        if (mx <= x) smx = mx, mx = x;
        else smx = max(smx, x);
    }
    printf("%d\n", smx);
    return 0;
}