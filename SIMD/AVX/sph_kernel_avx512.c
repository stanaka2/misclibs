/*
gcc -O3 -mavx512f sph_kernel_avx512.c
*/

#include <x86intrin.h>
#include <math.h>

#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))

static inline __m512 _mm512_WendlandQuinticC4_ps(const __m512 _rad, const __m512 _h)
{
  /*
  a = 495.0 / (256.0 * M_PI * CUBE(h));
  q = r / h;
  w = 0.0;
  if(q <= 2) w = SQR(CUBE(1.0 - 0.5 * q)) * (SQR(q) * 35.0 / 12.0 + 3.0 * q + 1.0);
  return a * w;
 */

  __m512 _zero = _mm512_set1_ps(0.0f);
  __m512 _one = _mm512_set1_ps(1.0f);
  __m512 _two = _mm512_set1_ps(2.0f);
  __m512 _tmp1, _tmp2, _tmp3;
  _tmp1 = _mm512_mul_ps(_mm512_set1_ps(M_PI), _mm512_mul_ps(_h, _mm512_mul_ps(_h, _h)));
  __m512 _a = _mm512_div_ps(_mm512_set1_ps(495.0f), _mm512_mul_ps(_mm512_set1_ps(256.0f), _tmp1));
  __m512 _q = _mm512_div_ps(_rad, _h);

  _tmp1 = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), _q, _one);  // 1.0 - 0.5 * q
  _tmp1 = _mm512_mul_ps(_tmp1, _mm512_mul_ps(_tmp1, _tmp1)); // CUBE(1.0 - 0.5 * q)
  _tmp1 = _mm512_mul_ps(_tmp1, _tmp1);                       // SQR(CUBE(1.0 - 0.5 * q))

  _tmp3 = _mm512_set1_ps(35.0f / 12.0f);
  _tmp2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(3.0f), _one);
  _tmp2 = _mm512_fmadd_ps(_mm512_mul_ps(_q, _q), _tmp3, _tmp2);

  _tmp1 = _mm512_mul_ps(_tmp1, _tmp2);
  _tmp1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_q, _two, _MM_CMPINT_LE), _zero, _tmp1);
  return _mm512_mul_ps(_a, _tmp1);
}

static inline __m512 _mm512_WendlandQuinticC4_dwdq_ps(const __m512 _rad, const __m512 _h)
{
  /*
  a = 495.0 / (256.0 * M_PI * CUBE(h));
  q = r / h;
  dwdq = 0.0;
  if(q <= 2) dwdq = (SQR(SQR(1.0 - 0.5 * q)) * (1.0 - 0.5 * q)) * (-14.0 / 3.0) * q * (1.0 + 2.5 * q);
  return a * dwdq;
  */
  __m512 _zero = _mm512_set1_ps(0.0f);
  __m512 _one = _mm512_set1_ps(1.0f);
  __m512 _two = _mm512_set1_ps(2.0f);
  __m512 _tmp1, _tmp2, _tmp3;
  _tmp1 = _mm512_mul_ps(_mm512_set1_ps(M_PI), _mm512_mul_ps(_h, _mm512_mul_ps(_h, _h)));
  __m512 _a = _mm512_div_ps(_mm512_set1_ps(495.0f), _mm512_mul_ps(_mm512_set1_ps(256.0f), _tmp1));
  __m512 _q = _mm512_div_ps(_rad, _h);

  _tmp1 = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), _q, _one); // 1.0 - 0.5 * q
  _tmp2 = _mm512_mul_ps(_tmp1, _mm512_mul_ps(_tmp1, _tmp1));
  _tmp1 = _mm512_mul_ps(_mm512_mul_ps(_tmp1, _tmp1), _tmp2); // SQR(1.0 - 0.5 * q)*CUBE(1.0 - 0.5 * q)

  _tmp3 = _mm512_fmadd_ps(_mm512_set1_ps(2.5f), _q, _one); // 1.0 + 2.5 * q
  _tmp2 = _mm512_set1_ps(-14.0f / 3.0f);
  _tmp2 = _mm512_mul_ps(_mm512_mul_ps(_tmp2, _q), _tmp3); // (-14.0 / 3.0) * q * (1.0 + 2.5 * q)

  _tmp1 = _mm512_mul_ps(_tmp1, _tmp2);
  _tmp1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_q, _two, _MM_CMPINT_LE), _zero, _tmp1);
  return _mm512_mul_ps(_a, _tmp1);
}

inline float WendlandQuinticC4(float r, const float h)
{
  float a = 495.0 / (256.0 * M_PI * CUBE(h));
  float q = r / h;
  float w = 0.0;
  if(q <= 2) w = SQR(CUBE(1.0 - 0.5 * q)) * (SQR(q) * 35.0 / 12.0 + 3.0 * q + 1.0);
  return a * w;
}

inline float WendlandQuinticC4_dwdq(float r, const float h)
{
  float a = 495.0 / (256.0 * M_PI * CUBE(h));
  float q = r / h;
  float dwdq = 0.0;
  if(q <= 2) dwdq = (SQR(SQR(1.0 - 0.5 * q)) * (1.0 - 0.5 * q)) * (-14.0 / 3.0) * q * (1.0 + 2.5 * q);
  return a * dwdq;
}

#if 1
#include <stdio.h>

int main(int argc, char **argv)
{
  const int n = 64;
  float rad[n] __attribute__((aligned(64)));
  float h[n] __attribute__((aligned(64)));
  float ret[n], retd[n];

  float ret_simd[n] __attribute__((aligned(64)));
  float retd_simd[n] __attribute__((aligned(64)));

  for(int i = 0; i < n; i++) {
    rad[i] = 0.05 * i;
    h[i] = 1.0;
  }

  for(int i = 0; i < n; i++) {
    ret[i] = WendlandQuinticC4(rad[i], h[i]);
    retd[i] = WendlandQuinticC4_dwdq(rad[i], h[i]);
  }

  for(int i = 0; i < n; i += 16) {

    __m512 _rad = _mm512_load_ps(&rad[i]);
    __m512 _h = _mm512_load_ps(&h[i]);
    __m512 _ret = _mm512_WendlandQuinticC4_ps(_rad, _h);
    __m512 _retd = _mm512_WendlandQuinticC4_dwdq_ps(_rad, _h);
    _mm512_store_ps(&ret_simd[i], _ret);
    _mm512_store_ps(&retd_simd[i], _retd);
  }

  for(int i = 0; i < n; i++) {
    printf("%d %g %g : %g %g : %g %g : %g %g\n", i, rad[i], h[i], ret[i], ret_simd[i], retd[i], retd_simd[i],
           ret[i] / ret_simd[i], retd[i] / retd_simd[i]);
  }

  return 0;
}

#endif
