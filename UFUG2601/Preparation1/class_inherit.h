#ifndef __CLASS_INHERIT_H__
#define __CLASS_INHERIT_H__
class Person {
protected:
    int Age, Region;
    int DNA;
public:
    Person();
    virtual int GetAge();
    void SetAge(int Age);
    int GetRegion();
    void SetRegion(int Region);
};

class Female : public Person {
public:
    int GetAge() override;
    int GetRegion();
};

#endif