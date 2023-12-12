#ifndef MY_COLLECTION_H
#define MY_COLLECTION_H

#include "Collections.hpp"
#include <stdexcept>

template <typename E>
class ArrayList : public List<E> {
    using size_t = unsigned long;
    // TODO
private:
    E *data;
    size_t capacity, length;

public:
    ArrayList();

    size_t size() const override;

    bool isEmpty() const override;

    bool contains(const E& element) const override;

    E get(int index) const;

    void clear();

    void add(const E& element) override;
    void add(int index, const E& element) override;

    bool remove(const E& element) override;
    E removeIndex(int index) override;

    int indexOf(const E& element) override;
};


template <typename E>
class LinkedList : public List<E> {
    // TODO
};






#endif // COLLECTION_H

