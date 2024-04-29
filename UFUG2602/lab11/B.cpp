#include <bits/stdc++.h>

using namespace std;

const int maxn = 1e6 + 5, maxm = 2e6 + 5;

struct Edge {
    int v, nex;
    Edge(int v = 0, int nex = 0) : v(v), nex(nex) {}
} E[maxm << 1];
int hd[maxn], tote;
void addedge(int u, int v) {
    E[++tote] = Edge(v, hd[u]), hd[u] = tote;
    E[++tote] = Edge(u, hd[v]), hd[v] = tote;
}

int dis[maxn], vis[maxn], cnt[maxn];
queue<int> q;

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "aw", stdout);
    int n, m; scanf("%d%d", &n, &m);
    for (int i = 1; i <= m; i++) {
        int u, v; scanf("%d%d", &u, &v);
        addedge(u, v);
    }
    q.push(1);
    memset(dis, -1, sizeof(dis));
    dis[1] = 0, cnt[1] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        vis[u] = 0;
        for (int i = hd[u]; i; i = E[i].nex) {
            int v = E[i].v;
            if (dis[v] == -1 || dis[v] > dis[u] + 1) {
                dis[v] = dis[u] + 1, cnt[v] = cnt[u];
                if (!vis[v]) q.push(v), vis[v] = 1;
            }
            else if (dis[v] == dis[u] + 1) (cnt[v] += cnt[u]) %= 100003;
        }
    }
    for (int i = 1; i <= n; i++) printf("%d\n", cnt[i]);
    return 0;
}