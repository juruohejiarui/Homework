#include <bits/stdc++.h>

using namespace std;

int vis[15], a[3];
void dfs(int x) {
    if (x >= 3) {
        a[1] = (a[0] << 1), a[2] = a[0] * 3;
        int s = 0, succ = true;
        for (int i = 0; i < 3; i++) {
            if (a[i] < 100 || a[i] > 999) { succ = false; break; }
            for (int d = 0, ai = a[i]; d < 3; d++, ai /= 10) {
                if (ai % 10 == 0) { succ = false; break; }
                if (s & (1 << (ai % 10 - 1))) { succ = false; break; }
                s |= (1 << (ai % 10 - 1));
            }
            if (!succ) break;
        }
        if (succ) printf("%d %d %d\n", a[0], a[1], a[2]);
        return ;
    }
    for (int d = 1; d <= 9; d++) if (!vis[d]) {
        vis[d] = 1;
        int id = x / 3, lsta = a[id];
        a[id] = a[id] * 10 + d;
        dfs(x + 1);
        a[id] = lsta, vis[d] = 0;
    }
}

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    dfs(0);
    return 0;
}