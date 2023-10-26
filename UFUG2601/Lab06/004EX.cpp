#include <bits/stdc++.h>

using namespace std;

const int maxn = 505, mv[5][2] = {{0,0}, {0,1}, {0,-1}, {1,0}, {-1,0}};
struct Point { int x, y; Point(int x, int y) : x(x), y(y) { } };
vector<Point> ls;
queue<Point> q;
int n, m, vis[maxn][maxn], ans[maxn][maxn];
char mp[maxn][maxn];

void bfs(int x, int y) {
    int cnt = 0;
    ls.clear();
    q.push(Point(x, y));
    while (!q.empty()) {
        auto u = q.front(); q.pop();
        vis[u.x][u.y] = 1;
        if (mp[u.x][u.y] == 'o') 
            cnt++, ls.push_back(u);
        for (int i = 1; i <= 4; i++) {
            int nx = u.x + mv[i][0], ny = u.y + mv[i][1];
            if (nx < 1 || ny < 1 || nx > n || ny > m || vis[nx][ny] || mp[nx][ny] == 'w')
                continue;
            q.push(Point(nx, ny)), vis[nx][ny] = 1;
        }
    }
    for (auto p : ls) ans[p.x][p.y] = cnt;
}
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            char ch = getchar();
            while (ch != 'w' && ch != 'x' && ch != 'o') ch = getchar();
            mp[i][j] = ch;
        }
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) if (!vis[i][j] && mp[i][j] != 'w')
            bfs(i, j);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (mp[i][j] == 'o') printf("%d", ans[i][j]);
            else putchar(mp[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
    return 0;
}