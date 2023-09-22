#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    LL a, b; string str;
    cin >> str >> a >> b;
    if (str == "ADD") printf("%lld\n", a + b);
    else if (str == "SUB") printf("%lld\n", a - b);
    else if (str == "RSF") printf("%lld\n", (a >> b));
    else if (str == "LSF") printf("%lld\n", (a << b));
    return 0;
}