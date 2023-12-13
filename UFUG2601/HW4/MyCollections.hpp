#ifndef MY_COLLECTION_H
#define MY_COLLECTION_H

#include "Collections.hpp"
#include <stdexcept>
#include <algorithm>

template <typename E>
class ArrayList : public List<E> {
    // TODO
private:
    E *data;
    int capacity, length;

public:
    ArrayList() {
        length = capacity = 0;
        data = nullptr;
    }
    ~ArrayList() {
        if (data != nullptr) delete[] data;
    }

    int size() const override {
        return length;
    }
    int getCapacity() const { return capacity; }

    bool isEmpty() const override {
        return length == 0;
    }

    bool contains(const E& element) const override {
        for (int i = 0; i < length; i++) if (data[i] == element) return true;
        return false;
}

    E get(int index) const override {
        if (index >= length) throw std::out_of_range("the index is out of range");
        return data[index];
    }


    void clear() {
        length = capacity = 0;
        delete[] data;
        data = nullptr;
    }

    void add(const E& element) override {
        if (length == capacity) {
            int newCapacity = std::max(1, capacity) << 1;
            E *newData = new E[newCapacity];
            for (int i = 0; i < length; i++) newData[i] = data[i];
            delete[] data;
            data = newData, capacity = newCapacity;
        }
        data[length++] = element;
    }
    void add(int index, const E& element) override {
        if (index > size()) throw std::out_of_range("the index is out of range");
        if (length == capacity) {
            int newCapacity = std::max(1, capacity) << 1;
            E *newData = new E[newCapacity];
            for (int i = 0; i < index; i++) newData[i] = data[i];
            newData[index] = element;
            for (int i = index; i < length; i++) newData[i + 1] = data[i];
            delete[] data;
            data = newData, capacity = newCapacity;
        } else {
            for (int i = length - 1; i >= index; i--) data[i + 1] = data[i];
            data[index] = element;
        }
        length++;
    }

    bool remove(const E& element) override {
        int pos = length;
        for (int i = 0; i < length; i++) if (data[i] == element) {
            pos = i;
            break;
        }
        if (pos == length) return false;
        removeIndex(pos);
        return true;
    }
    E removeIndex(int index) override {
        if (index >= length) throw std::out_of_range("the index is out of range");
        E res = data[index];
        if (length - 1 <= (capacity >> 2)) {
            E *newData = new E[capacity >> 1];
            for (int i = 0; i < index; i++) newData[i] = data[i];
            for (int i = index + 1; i < length; i++) newData[i - 1] = data[i];
            delete[] data;
            data = newData;
            length--, capacity >>= 1;
        } else {
            for (int i = index + 1; i < length; i++) data[i - 1] = data[i];
            length--;
        }
        return res;
    }

    int indexOf(const E& element) const override {
        for (int i = 0; i < length; i++) if (data[i] == element) return i;
        return -1;
    }
};

template <typename E>
class LinkedList : public List<E> {
    // TODO
private:
    class Element {
        Element *prev, *next;
        T content;
    } *start, *end;
    
    LinkedList() {
        start = new Element, end = new Element;
    }
    ~LinkedList() {
        
    }
};

#endif // COLLECTION_H

