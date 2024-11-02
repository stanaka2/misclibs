#pragma once

#include <iostream>

template <typename T>
void printVec(const T &vec, const std::string &label)
{
  std::cout << label << ": ";
  for(const auto &v : vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}
