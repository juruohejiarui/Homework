#include <algorithm>
#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cmath>

using namespace std;

const int maxn = 25;
char mp[maxn][maxn];
int n, m, q;

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    scanf("%d%d%d", &n, &m, &q);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) mp[i][j] = '.';
    while (q--) {
        int x, y, mx = 0, my = 0; char c, d;
        scanf("%d%d", &x, &y);
        c = getchar();
        while (c < 'a' || c > 'z') c = getchar();
        d = getchar();
        while (d != 'L' && d != 'U' && d != 'R' && d != 'D') d = getchar();
        switch(d) {
            case 'U': mx = -1; break;
            case 'D': mx = 1; break;
            case 'L': my = -1; break;
            case 'R': my = 1; break;
        }
        for (; 1 <= x && x <= n && 1 <= y && y <= m; x += mx, y += my)
            mp[x][y] = c;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) putchar(mp[i][j]);
        putchar('\n');
    }
    return 0;
}