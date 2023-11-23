#include <bits/stdc++.h>

using namespace std;

const int maxn = 1e5 + 5;
char str[maxn]; int len, p1, p2, p3;

inline char is_number(char ch) { return ch >= '0' && ch <= '9'; }
inline char is_letter(char ch) { return ch >= 'a' && ch <= 'z'; }
void print_rep(char ch, int x) { while (x--) putchar(ch); }

int main()
{
	scanf("%d%d%d\n%s", &p1, &p2, &p3, str + 1);
	len = strlen(str + 1);
	putchar(str[1]);
	for (int i = 2; i <= len; i++) {
		if (str[i] == '-') {
			if (i == len) putchar('-');
			else {
				char lst = str[i - 1], nxt = str[i + 1];
				if ((is_number(lst) && is_number(nxt)) || (is_letter(lst) && is_letter(lst))) {
					if (lst < nxt) {
						if (p3 == 1) for (char ch = lst + 1; ch < nxt; ch++)
							print_rep((p1 == 3 ? '*' : (is_number(ch) ? ch : (p1 == 2 ? ch - 'a' + 'A' : ch))), p2);
						else for (char ch = nxt - 1; ch > lst; ch--)
							print_rep((p1 == 3 ? '*' : (is_number(ch) ? ch : (p1 == 2 ? ch - 'a' + 'A' : ch))), p2);
					} else putchar('-');
				} else putchar('-');
			}
		} else putchar(str[i]);
	}
	return 0;
}