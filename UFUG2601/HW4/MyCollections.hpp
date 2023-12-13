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
class LinkedListElement {
public:
    LinkedListElement *prev, *next;
    E data;
    LinkedListElement() { prev = next = nullptr; }
};

template <typename E>
class LinkedList : public List<E> {
    // TODO
private:
    LinkedListElement<E> *start, *end;
    int length;
public:
    LinkedList() {
        start = new LinkedListElement<E>, end = new LinkedListElement<E>;
        start->next = end, end->prev = start;
        length = 0;
    }
    ~LinkedList() {
        for (LinkedListElement<E> *ele = start, *nxt = ele; ele != end; ele = nxt) {
            nxt = ele->next;
            delete ele;
        }
        delete end;
    }
    int size() const override { return length; }

    bool isEmpty() const override { return length == 0; }

    bool contains(const E& element) const override {
        for (LinkedListElement<E> *ele = start->next; ele != end; ele = ele->next)
            if (ele->data == element) return true;
        return false;
    }

    E get(int index) const override {
        if (index >= length) throw std::out_of_range("the index is out of range");
        LinkedListElement<E> *ele;
        for (ele = start->next; index; ele = ele->next, index--) ;
        return ele->data;
    }

    void clear() {
        length = 0;
        for (LinkedListElement<E> *ele = start->next, *nxt = ele; ele != end; ele = nxt) {
            nxt = ele->next;
            delete ele;
        }
        start->next = end, end->prev = start;
    }

    void add(const E& element) override {
        LinkedListElement<E> *newEle = new LinkedListElement<E>;
        newEle->prev = end->prev;
        newEle->next = end;
        newEle->prev->next = newEle;
        end->prev = newEle;
        newEle->data = element;
        length++;
    }

    void add(int index, const E& element) override {
        if (index >= length) throw std::out_of_range("the index is out of range");
        length++;
        LinkedListElement<E> *ele = new LinkedListElement<E>, *pos;
        ele->data = element;
        for (pos = start->next; index; pos = pos->next, index--) ;
        ele->next = pos, ele->prev = pos->prev, pos->prev = ele, ele->prev->next = ele;
    }

    bool remove(const E& element) override {
        for (LinkedListElement<E> *ele = start->next; ele != end; ele = ele->next) 
            if (ele->data == element) {
                ele->prev->next = ele->next, ele->next->prev = ele->prev;
                delete ele;
                length--;
                return true;
            }
        return false;
    }

    E removeIndex(int index) override {
        if (index >= length) throw std::out_of_range("the index is out of range");
        LinkedListElement<E> *ele;
        E res;
        if (index >= length) for (ele = end->prev, index = length - index - 1; index; ele = ele->prev, index--) ;
        else for (ele = start->next; index; index--, ele = ele->next) ;
        length--;
        res = ele->data, ele->prev->next = ele->next, ele->next->prev = ele->prev;
        delete ele;
        return res;
    }

    int indexOf(const E& element) const override {
        for (auto ele = start->next, index = 0; ele != end; ele = ele->next, index++)
            if (ele->data == element) return index;
        return -1;
    }
};

#endif // COLLECTION_H

