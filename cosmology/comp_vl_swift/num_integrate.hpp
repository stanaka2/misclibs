#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <algorithm>

namespace num
{

template <class F, class T>
T quad(F &&f, T a, T b, T eps = static_cast<T>(1e-8), int max_depth = 20)
{
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  auto simpson = [](T fa, T fm, T fb, T h) noexcept { return (h / T(6)) * (fa + T(4) * fm + fb); };

  auto eval = [&](T x) -> T {
    using std::invoke;
    return static_cast<T>(invoke(std::forward<F>(f), x));
  };

  const T fa = eval(a);
  const T fb = eval(b);
  const T m = (a + b) / T(2);
  const T fm = eval(m);
  const T Sab = (b - a) / T(6) * (fa + T(4) * fm + fb);

  std::function<T(T, T, T, T, T, T, int)> rec = [&](T a, T b, T fa, T fm, T fb, T Sab, int depth) -> T {
    const T m = (a + b) / T(2);
    const T h = (b - a) / T(2);
    const T lm = (a + m) / T(2);
    const T rm = (m + b) / T(2);
    const T flm = eval(lm);
    const T frm = eval(rm);
    const T Sl = simpson(fa, flm, fm, h);
    const T Sr = simpson(fm, frm, fb, h);
    const T S2 = Sl + Sr;
    if(depth <= 0 || std::abs(S2 - Sab) <= T(15) * eps) {
      return S2 + (S2 - Sab) / T(15);
    }
    return rec(a, m, fa, flm, fm, Sl, depth - 1) + rec(m, b, fm, frm, fb, Sr, depth - 1);
  };

  return rec(a, b, fa, fm, fb, Sab, max_depth);
}

template <class F, class T>
T brentq(F &&f, T ax, T bx, T tol = static_cast<T>(1e-10), int maxit = 100)
{
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  auto eval = [&](T x) -> T {
    using std::invoke;
    return static_cast<T>(invoke(std::forward<F>(f), x));
  };

  T a = ax, b = bx, fa = eval(a), fb = eval(b);
  if(fa == T(0)) return a;
  if(fb == T(0)) return b;
  if(fa * fb > T(0)) throw std::runtime_error("brentq: root not bracketed");

  T c = a, fc = fa, d = b - a, e = d;
  for(int it = 0; it < maxit; ++it) {
    if(std::abs(fc) < std::abs(fb)) {
      std::swap(a, b);
      std::swap(fa, fb);
      std::swap(c, a);
      std::swap(fc, fa);
    }
    const T tol1 = T(2) * std::numeric_limits<T>::epsilon() * std::abs(b) + tol / T(2);
    const T xm = (c - b) / T(2);
    if(std::abs(xm) <= tol1 || fb == T(0)) return b;

    if(std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
      T s = fb / fa, p, q;
      if(a == c) {
        p = T(2) * xm * s;
        q = T(1) - s;
      } else {
        T q1 = fa / fc, r = fb / fc;
        p = s * (T(2) * xm * q1 * (q1 - r) - (b - a) * (r - T(1)));
        q = (q1 - T(1)) * (r - T(1)) * (s - T(1));
      }
      if(p > 0) q = -q;
      p = std::abs(p);
      if(T(2) * p < std::min(T(3) * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }

    a = b;
    fa = fb;
    b += (std::abs(d) > tol1) ? d : ((xm > 0) ? tol1 : -tol1);
    fb = eval(b);
    if((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
      c = a;
      fc = fa;
      e = d = b - a;
    }
  }
  throw std::runtime_error("brentq: max iterations exceeded");
}

template <class F, class T>
auto invert_mono_increasing(F &&g, T a_lo, T a_hi, T tol = static_cast<T>(1e-10), int maxit = 100)
{
  return [=, func = std::forward<F>(g)](T y) {
    auto Froot = [&](T a) { return static_cast<T>(std::invoke(func, a) - y); };
    T lo = a_lo, hi = a_hi;
    int guard = 0;
    while(Froot(lo) > 0 && guard++ < 60) lo = std::max(lo / T(2), std::numeric_limits<T>::lowest());
    guard = 0;
    while(Froot(hi) < 0 && guard++ < 60) hi = std::min(hi * T(2), std::numeric_limits<T>::max() / T(4));
    return brentq(Froot, lo, hi, tol, maxit);
  };
}

} // namespace num
