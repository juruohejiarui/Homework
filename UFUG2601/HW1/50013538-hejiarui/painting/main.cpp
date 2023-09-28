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
        int x, y, mx = 0, my = 0; char col, direction;
        scanf("%d%d", &x, &y);
        col = getchar();
        while (col < 'a' || col > 'z') col = getchar();
        direction = getchar();
        while (direction != 'L' && direction != 'U' && direction != 'R' && direction != 'D') 
            direction = getchar();
        switch(direction) {
            case 'U': mx = -1; break;
            case 'D': mx = 1; break;
            case 'L': my = -1; break;
            case 'R': my = 1; break;
        }
        for (; 1 <= x && x <= n && 1 <= y && y <= m; x += mx, y += my)
            mp[x][y] = col;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) putchar(mp[i][j]);
        putchar('\n');
    }
    return 0;
}