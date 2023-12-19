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


int main() {
    // Insert your test code here. 
    // You can use multiple functions similar to the example provided. 
    // Thanks to inheritance, these functions can work with both ArrayList<T> and LinkedList<T>! 
    

    // e.g., 
    LinkedList<int> list;
    test_list0(list);

    // ... 

    


    return 0;
}