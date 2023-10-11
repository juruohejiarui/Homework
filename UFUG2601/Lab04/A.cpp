#include <bits/stdc++.h>

using namespace std;

const int maxn = 1e6 + 5;
int a[maxn];
void solve() {
    int n, st, ed;
    scanf("%d%d%d", &n, &st, &ed);
    for (int i = 0; i < n; i++) scanf("%d", &a[i]);
    int fl_u = 1, fl_d = 1;
    for (int i = st; i < ed; i++) {
        if (a[i] < a[i + 1]) fl_d = 0;
        else if (a[i] > a[i + 1]) fl_u = 0;
    }
    if (fl_u ^ fl_d)
        printf("%s\n", (fl_u ? "UP" : "DOWN"));
    else printf("UP&DOWN\n");
}
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int T; scanf("%d", &T);
    while (T--) solve();
    return 0;
}