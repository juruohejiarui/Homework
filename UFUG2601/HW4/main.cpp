#include "Collections.hpp"
#include "MyCollections.hpp"

#include <stdexcept>
#include <iostream>

template<typename T>
void test_list0(List<T>& list) {
    list.add(5);
    list.add(6);
    list.add(7);
    // ... 
}


int main() {
    // Insert your test code here. 
    // You can use multiple functions similar to the example provided. 
    // Thanks to inheritance, these functions can work with both ArrayList<T> and LinkedList<T>! 
    

    // e.g., 
    ArrayList<int> list;
    test_list0(list);

    // ... 

    


    return 0;
}