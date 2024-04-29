#include <bits/stdc++.h>

using namespace std;

const int maxn = 1e6 + 5;

int readint() {
    int x = 0, f = 1; char ch = getchar();
    while (!isdigit(ch)) { if (ch == '-') f = -1; ch = getchar(); }
    while (isdigit(ch)) { x = x * 10 + ch - '0'; ch = getchar(); }
    return x * f;
}

vector<int> G[maxn];
int deg[maxn], vis[maxn];
queue<int> q;

int ans[maxn], ansl;

void dfs(int st) {
    if (vis[st]) return ;
    vis[st] = 1;
    ans[++ansl] = st;
    for (int v : G[st]) {
        if (vis[v]) continue;
        dfs(v);
    }
}
void bfs(int i) {
    q.push(i), vis[i] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        ans[++ansl] = u;
        for (int i = 0; i < G[u].size(); i++) {
            int v = G[u][i];
            if (!vis[v]) q.push(v), vis[v] = 1;
        }
    }
}

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int n, m; scanf("%d%d", &n, &m);
    for (int i = 1; i <= m; i++) {
        int u = readint(), v = readint();
        G[u].push_back(v);
    }
    for (int i = 1; i <= n; i++) sort(G[i].begin(), G[i].end());
    for (int i = 1; i <= n; i++) if (!vis[i]) dfs(i);
    for (int i = 1; i <= ansl; i++) printf("%d%c", ans[i], (i == ansl ? '\n' : ' '));
    memset(vis, 0, sizeof(vis)), ansl = 0;
    for (int i = 1; i <= n; i++) if (!vis[i]) bfs(i);
    for (int i = 1; i <= ansl; i++) printf("%d%c", ans[i], (i == ansl ? '\n' : ' '));
    return 0;
}