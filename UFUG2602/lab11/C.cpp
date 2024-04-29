#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
const int maxn = 1e5 + 5, maxm = 5e6 + 5;

int readint() {
    int x = 0, f = 1; char ch = getchar();
    while (!isdigit(ch)) { if (ch == '-') f = -1; ch = getchar(); }
    while (isdigit(ch)) { x = x * 10 + ch - '0'; ch = getchar(); }
    return x * f;
}

struct Edge {
    int v, nex;
    Edge(int v = 0, int nex = 0) : v(v), nex(nex) {}
} E[maxm];
int hd[maxn], tote;
void addedge(int u, int v) {
    E[++tote] = Edge(v, hd[u]), hd[u] = tote;
}

LL w[maxn], f[maxn];

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int n = readint();
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    int m = readint();
    for (int i = 1; i <= m; i++) {
        int u = readint(), v = readint();
        addedge(v, u);
    }
    for (int i = 1; i <= n; i++) {
        f[i] = w[i];
        for (int j = hd[i]; j; j = E[j].nex) {
            int v = E[j].v;
            if (v < i) f[i] = max(f[i], f[v] + w[i]);
        }
    }
    LL ans = 0;
    for (int i = 1; i <= n; i++) ans = max(ans, f[i]);
    printf("%lld\n", ans);
}