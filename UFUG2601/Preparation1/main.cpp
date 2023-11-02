#include "class_inherit.h"
#include <cstdio>

int main() {
    Female *female = new Female();
    Person *person = female;
    person->SetAge(24);
    printf("Age : %d %d\nRegion : %d %d\n", person->GetAge(), female->GetAge(), person->GetRegion(), female->GetRegion());
} 