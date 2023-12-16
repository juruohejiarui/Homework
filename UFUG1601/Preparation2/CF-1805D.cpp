#include <bits/stdc++.h>

using namespace std;

const int maxn = 1e5 + 5;
struct Edge {
    int v, nex;
    Edge(int v = 0, int nex = 0) : v(v), nex(nex) {}
} E[maxn << 1];
int hd[maxn], tote;
void addedge(int u, int v) {
    E[++tote] = Edge(v, hd[u]), hd[u] = tote;
    E[++tote] = Edge(u, hd[v]), hd[v] = tote;
}

int dis[maxn], dis2[maxn], cnt[maxn], ans[maxn], mxd, mxdp;
void dfs(int *dis, int u, int fa) {
    dis[u] = dis[fa] + 1;
    if (mxd < dis[u]) mxd = dis[mxdp = u];
    for (int i = hd[u]; i; i = E[i].nex) {
        int v = E[i].v;
        if (v == fa) continue;
        dfs(dis, v, u);
    }
}
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int n, p1, p2; scanf("%d", &n);
    for (int i = 1; i < n; i++) {
        int u, v; scanf("%d%d", &u, &v);
        addedge(u, v);
    }
    dfs(dis, 1, 0), p1 = mxdp, mxd = 0;
    dfs(dis, p1, 0), p2 = mxdp, mxd = 0;
    dfs(dis2, p2, 0);

    for (int i = 1; i <= n; i++) cnt[max(dis[i], dis2[i]) - 1]++;
    ans[n] = n;
    for (int i = n - 1; i >= 1; i--) 
        if (!cnt[i]) ans[i] = ans[i + 1];
        else if (ans[i + 1] == n) ans[i] = ans[i + 1] - cnt[i] + 1;
        else ans[i] = ans[i + 1] - cnt[i];
    for (int i = 1; i <= n; i++) printf("%d ", ans[i]);
    return 0;
}