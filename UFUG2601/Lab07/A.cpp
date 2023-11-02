#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
struct Rect {
    int x1, x2, y1, y2;
    Rect(int x1 = 0, int y1 = 0, int x2 = 0, int y2 = 0) : x1(x1), y1(y1), x2(x2), y2(y2) {}
    LL CalcS() {
        return 1ll * max(0, (x2 - x1)) * max(0, (y2 - y1));
    }
    static Rect GetIntersect(const Rect &a, const Rect &b) {
        return Rect(max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2));
    }
} a, b;

int main() {
    freopen("test.in", "r", stdin);
    freopen("test.out", "w", stdout);
    scanf("%d%d%d%d", &a.x1, &a.y1, &a.x2, &a.y2);
    scanf("%d%d%d%d", &b.x1, &b.y1, &b.x2, &b.y2);
    printf("%lld\n", Rect::GetIntersect(a, b).CalcS());
    return 0;
}