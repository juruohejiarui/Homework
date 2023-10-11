#include <bits/stdc++.h>

using namespace std;

set<int> a, b, c;

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    int n1, n2; scanf("%d%d", &n1, &n2);
    for (int i = 1; i <= n1; i++) { int x; scanf("%d", &x), a.insert(x), c.insert(x); }
    for (int i = 1; i <= n2; i++) { int x; scanf("%d", &x), b.insert(x), c.insert(x); }
    if (n1 + n2 == 0) printf("None");
    else for (auto i = c.begin(); i != c.end(); i++) printf("%d ", *i);
    putchar('\n');
    c.clear();
    auto i = a.begin(), j = b.begin();
    while (i != a.end()) {
        while (*j < *i && j != b.end()) j++;
        if (j == b.end()) break;
        if (*j == *i) c.insert(*j);
        i++;
    }
    if (c.size() == 0) printf("None\n");
    else for (i = c.begin(); i != c.end(); i++) printf("%d ", *i);
    return 0;
}