#include "Collections.hpp"
#include "MyCollections.hpp"

#include <iostream>
#include <cstdlib>

template<typename T>
void test_list0(LinkedList<T>& list) {
    list.add(5);
    list.add(6);
    list.add(7);
    list.add(998244353);
    
    for (int i = 0; i < list.size(); i++) std::cout << list.get(i) << ' ';
    putchar('\n');

    list.remove(7);
    for (int i = 0; i < list.size(); i++) std::cout << list.get(i) << ' ';
    putchar('\n');

    list.add(2, 114514);
    list.removeIndex(2);
    for (int i = 0; i < list.size(); i++) std::cout << list.get(i) << ' ';
    putchar('\n');
    
    for (int i = 0; i < 1000000; i++) list.add(i * 2);
    // for (int i = 0; i < list.size(); i++) std::cout << list.get(i) << ' ';
    // putchar('\n');
    int c = list.size() * 3 / 4 + 4;
    // printf("capacity = %d\n", list.getCapacity());
    srand(time(NULL));
    while (c--) {
        int x = rand() % 2;
        int t = list.removeIndex(x ? list.size() - 1 - rand() % 30 : 0 + rand() % 30);
        // printf("remove index = %d, t = %d\n", x, t);
    }
    for (int i = 0; i < 1000; i++) std::cout << list.get(i) << ' ';
    putchar('\n');
    list.clear();
}

void test1() {
	ArrayList<int> ls;
	std::vector<int> arr;
	int Q = 10000;
	srand(time(NULL));
	while (Q--) {
		int op = rand() % 10;
        // printf("op = %d ", op);
        if (ls.size() != (int)arr.size()) {
            printf("neq size\n");
            break;
        }
		if (op == 0 || op > 8) {
			int x = rand() % 1000;
			ls.add(x), arr.push_back(x);
		} else if (op == 1) {
            int p;
            if (ls.size() == 0) p = 0;
            else p = rand() % 2 == 0 ? rand() % ls.size() : ls.size();
			int x = rand() % 1000; 
			ls.add(p, x), arr.insert(arr.begin() + p, x);
		} else if (op == 2 || op == 6) {
			if (arr.size() == 0) continue;
			ls.removeIndex(ls.size() - 1), arr.pop_back();
		} else if (op == 3) {
			if (arr.size() == 0) continue;
			int p = rand() % ls.size();
            ls.removeIndex(p), arr.erase(arr.begin() + p);
        } else if (op == 4) {
            if (arr.size() == 0) continue;
            int p = rand() % ls.size();
            printf("%c", (ls.indexOf(ls.get(p)) == std::find(arr.begin(), arr.end(), ls.get(p)) - arr.begin()) ? 'Y' : 'N');
        } else if (op == 5 || op == 7 || op == 8) {
            bool succ = true;
            for (int i = 0; i < ls.size(); i++)
                if (ls.get(i) != arr[i]) { succ = false; break; }
            printf("%c", (succ ? 'Y' : 'N'));
        }
    }
}

int main() {
    // Insert your test code here. 
    // You can use multiple functions similar to the example provided. 
    // Thanks to inheritance, these functions can work with both ArrayList<T> and LinkedList<T>! 
    

    // e.g., 
    LinkedList<int> list;
    test_list0(list);
    
    test1();
    // ... 

    


    return 0;
}
