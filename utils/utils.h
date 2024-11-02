#pragma once

#include <stdio.h>
#include <stdint.h>

void printVecf(const float *vec, const int size, const char *label)
{
  printf("%s :", label);
  for(int i = 0; i < size; i++) printf(" %g", vec[i]);
  printf("\n");
}

void printVecd(const double *vec, const int size, const char *label)
{
  printf("%s :", label);
  for(int i = 0; i < size; i++) printf(" %g", vec[i]);
  printf("\n");
}

void printVeci(const int *vec, const int size, const char *label)
{
  printf("%s :", label);
  for(int i = 0; i < size; i++) printf(" %d", vec[i]);
  printf("\n");
}

void printVecld(const int64_t *vec, const int size, const char *label)
{
  printf("%s :", label);
  for(int i = 0; i < size; i++) printf(" %ld", vec[i]);
  printf("\n");
}

#define printVec(vec, size, label) \
  _Generic((vec), float *: printVecf, double *: printVecd, int *: printVeci, int64_t *: printVecld)(vec, size, label)
