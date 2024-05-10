#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
const int maxn = 5e3 + 5;
struct Edge {
	int v; LL w;
	Edge(int v = 0, LL w = 0) : v(v), w(w) {}
};
vector<Edge> G[maxn];
LL mnd[maxn], cnt[maxn], inq[maxn];

bool spfa(int st, int limit) {
	memset(cnt, 0, sizeof(cnt)), memset(inq, 0, sizeof(inq)), memset(mnd, 0, sizeof(mnd));
	queue<int> q; q.push(st);
	cnt[st] = inq[st] = 1;
	while (!q.empty()) {
		int u = q.front(); q.pop();
		inq[u] = 0;
		for (const Edge &e : G[u]) {
			if (cnt[e.v] == 0 || mnd[e.v] > mnd[u] + e.w) {
				mnd[e.v] = mnd[u] + e.w;
				cnt[e.v]++;
				if (cnt[e.v] >= limit) return false;
				if (!inq[e.v]) q.push(e.v), inq[e.v] = 1;
			}
		}
	}
	return true;
}

int main() {
	// freopen("test.in", "r", stdin);
	// freopen("test.out", "w", stdout);
	int n, m; scanf("%d%d", &n, &m);
	for (int i = 1; i <= m; i++) {
		int u, v; LL w;
		scanf("%d%d%lld", &u, &v, &w);
		G[v].push_back(Edge(u, w));
	}
	for (int st = 1; st <= n; st++) if (!cnt[st]) {
		bool res = spfa(st, n + 1);
		if (!res) {
			printf("NO\n");
			return 0;
		}
	}
	printf("YES\n");
	return 0;
}