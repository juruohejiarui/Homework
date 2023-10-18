#include <bits/stdc++.h>

using namespace std;

const int maxn = 55;
int n, m, f[maxn][maxn];
char mp[maxn][maxn];

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    scanf("%d%d", &n, &m);
    int stx = 0, sty = 0, edx, edy;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            char ch = getchar();
            while (ch != 'x' && ch != 'o' && ch != 'w') ch = getchar();
            if (ch == 'o') {
                if (!stx) stx = i, sty = j;
                else edx = i, edy = j;
            }
            if (ch == 'w') mp[i][j] = 'w';
            else mp[i][j] = 'x';
        }
    f[stx][sty] = 1;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) if (mp[i][j] != 'w')
            f[i][j] |= f[i - 1][j] | f[i][j - 1];
    printf("%s\n", (f[edx][edy] ? "True" : "False"));
    return 0;
}