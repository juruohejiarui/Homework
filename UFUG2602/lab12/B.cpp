#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
const int maxn = 1e5 + 5, maxm = 2e5 + 5, maxk = (maxn << 1) + maxm;

struct Edge {
	int v; LL w;
	Edge(int v = 0, LL w = 0) : v(v), w(w) {}
};
vector<Edge> G[maxk];

LL dis[maxk], vis[maxk];
priority_queue< pair<LL, int> > pq;
int n, m; LL K;

void dij() {
	memset(dis, -1, sizeof(dis));
	pq.push(make_pair(0, 1));
	dis[1] = 0;
	while (!pq.empty()) {
		int u = pq.top().second; pq.pop();
		if (vis[u]) continue;
		vis[u] = 1;
		for (const Edge &e : G[u])
			if (dis[e.v] == -1 || dis[e.v] > dis[u] + e.w)
				dis[e.v] = dis[u] + e.w,
				pq.push(make_pair(-dis[e.v], e.v));
	}
}

map<LL, vector<int> > mp[maxn];
map<LL, int> grpId[maxn];
tuple<int, int, LL> edges[maxm];

void build(int e, int u, LL w) {
	G[n + e].push_back(Edge(u, 0));
	G[u].push_back(Edge(n + e, w));
	if (mp[u].count(K * w))
		G[n + e].push_back(Edge(grpId[u][K * w], 0));
}
int main() {
	// freopen("test.in", "r", stdin);
	// freopen("test.out", "w", stdout);
	scanf("%d%d%lld", &n, &m, &K);
	int grpIdC = 0;
	for (int i = 1; i <= m; i++) {
		int u, v; LL w; scanf("%d%d%lld", &u, &v, &w);
		edges[i] = make_tuple(u, v, w);
		if (!mp[u].count(w)) grpId[u][w] = n + m + (++grpIdC);
		if (!mp[v].count(w)) grpId[v][w] = n + m + (++grpIdC);
		mp[u][w].push_back(i);
		mp[v][w].push_back(i);
	}
	for (int i = 1; i <= m; i++) {
		int u = get<0>(edges[i]), v = get<1>(edges[i]); LL w = get<2>(edges[i]);
		build(i, u, w), build(i, v, w);
	}
	if (K > 0) {
		for (int i = 1; i <= n; i++) {
			for (const auto &pir : mp[i]) {
				int id = grpId[i][pir.first];
				const auto lst = pir.second;
				for (int x : lst) G[id].push_back(Edge(n + x, pir.first / K * (K - 1)));
			}
		}
	}
	
	dij();
	for (int i = 1; i <= n; i++) printf("%lld ", dis[i]);
	return 0;
}