#include <algorithm>
#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cmath>

using namespace std;

typedef long long LL;
LL getval(string& str, LL B) {
    LL res = 0;
    for (int i = 0; i < str.size(); i++)  res = res * B + (str[i] - '0');
    return res;
}
int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    string p, q, r;
    cin >> p >> q >> r;
    for (LL B = 2; B <= 16; B++) {
        LL vp = getval(p, B), vq = getval(q, B), vr = getval(r, B);
        if (vp * vq == vr) { cout << B << endl; return 0; }
    }
    cout << 0 << endl;
    return 0;
}