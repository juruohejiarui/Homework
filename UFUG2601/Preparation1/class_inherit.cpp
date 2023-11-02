#include "class_inherit.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

Person::Person() { Age = Region = 0; }

void Person::SetAge(int Age) { this->Age = Age; }
int Person::GetAge() { return Age; }
void Person::SetRegion(int Region) { this->Region = Region; }
int Person::GetRegion() { return Region; }

int Female::GetAge() { return std::min(18, Age); }

int Female::GetRegion() { return 1; }
