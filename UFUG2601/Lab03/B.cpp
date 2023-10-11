#include <bits/stdc++.h>

using namespace std;

void solve() {
    unsigned long long x, sqr; scanf("%llu", &x);
    sqr = sqrt(x);
    if (sqr * sqr == x) printf("Yes\n");
    else printf("No\n");
}
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int T; scanf("%d", &T);
    while (T--) solve();
    return 0;
}