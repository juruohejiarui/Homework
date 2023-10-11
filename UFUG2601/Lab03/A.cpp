#include <bits/stdc++.h>

using namespace std;

typedef unsigned long long ULL;

int main() {
    // freopen("test.in", "r", stdin);
    // freopen("test.out", "w", stdout);
    ULL A, B; scanf("%llu%llu", &A, &B);
    for (ULL i = A; i <= A + 8; i++) {
        for (ULL j = B; j <= B + 8; j++) 
            printf("%llu\t", i * j);
        printf("\n");
    }
    return 0;
}