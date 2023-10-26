#include <bits/stdc++.h>

using namespace std;

struct Info {
    double a, b, c;
    Info(double a, double b, double c) : a(a), b(b), c(c) {}
};
map<string, Info> mp;

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int n, m; scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) {
        string name; double a, b, c;
        cin >> name >> a >> b >> c;
        mp.insert(make_pair(name, Info(a, b, c)));
    }
    for (int i = 1; i <= m; i++) {
        string name;
        cin >> name;
        auto iter = mp.find(name);
        if (iter == mp.end()) printf("404 Not Found\n");
        else printf("%.0lf %.1lf %.1lf\n", iter->second.a, iter->second.b, iter->second.c);
    }
    return 0;
}