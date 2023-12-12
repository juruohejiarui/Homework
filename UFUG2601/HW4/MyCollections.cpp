#include "MyCollections.hpp"
#include <algorithm>

template <typename E>
size_t ArrayList<E>::size() const {
    return length;
}

template <typename E>
bool ArrayList<E>::isEmpty() const {
    return length == 0;
}

template <typename E>
bool ArrayList<E>::contains(const E &element) const {
    for (int i = 0; i < length; i++) if (data[i] == element) return true;
    return false;
}

template <typename E>
E ArrayList<E>::get(int index) const {
    if (index >= length) throw std::out_of_range();
    return data[index];
}

template <typename E>
void ArrayList<E>::clear() {
    length = capacity = 0;
    delete[] data;
}

template <typename E>
void ArrayList<E>::add(const E &element) {
    if (length == capacity) {
        size_t newCapacity = std::max(1lu, capacity) << 1;
        E *newData = new E[newCapacity];
        for (int i = 0; i < length; i++) newData[i] = data[i];
        delete[] data;
        data = newData;
    }
    data[length++] = element;
}

template <typename E>
void ArrayList<E>::add(int index, const E &element) {
    if (index > size()) throw std::out_of_range();
    if (length == capacity) {
        size_t newCapacity = std::max(1lu, capacity) << 1, offset = 0;
        E *newData = new E[newCapacity];
        for (int i = 0; i < index; i++) newData[i] = data[i];
        newData[index] = element;
        for (int i = index; i < length; i++) newData[i + 1] = data[i];
    } else {
        for (int i = length - 1; i >= index; i--) data[i + 1] = data[i];
        data[index] = element;
    }
    length++;
}

template <typename E>
bool ArrayList<E>::remove(const E &element) {
    size_t pos = length;
    for (int i = 0; i < length; i++) if (data[i] == element) {
        pos = i;
        break;
    }
    if (pos == length) return false;
    
    return true;
}
