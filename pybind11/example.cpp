/*
g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config
--extension-suffix`
*/

#include <pybind11/pybind11.h>

// 名前空間
namespace py = pybind11;

// C++で定義した関数
int add(int a, int b) { return a + b; }

// Pybind11のモジュール定義
PYBIND11_MODULE(example, m)
{
  m.doc() = "pybind11 example plugin";                                                // モジュールの説明 (オプション)
  m.def("add", &add, "A function that adds two numbers", py::arg("a"), py::arg("b")); // Pythonで関数として公開
}
