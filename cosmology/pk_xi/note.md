# note

## power spectrum of density fluctuations

### multipole component of $P(k)$

- $L_\ell(\mu)$ is Legendre polynomial

  ```math
  P(k,\mu) = \sum_\ell P_\ell(k) L_\ell(\mu) ,
  ```

- Power spectrum with multipole components

  ```math
  P_\ell(k) = \frac{2\ell+1}{2} \int^{1}_{-1} d\mu \, P(k,\mu) L_\ell(\mu) ,
  ```

- Here, $\mu = \cos\theta$ is an even function, so the implementation

  ```math
  P_\ell(k) = (2\ell+1) \int^{1}_{0} d\mu \, P(k,\mu) L_\ell(\mu) .
  ```

## correlation function of density fluctuations

### multipole component of $\xi(s)$

- $L_\ell(\mu)$ is Legendre polynomial

  ```math
  \xi(s,\mu) = \sum_\ell \xi_\ell(s) L_\ell(\mu) ,
  ```

- Correlation function with multipole components

  ```math
  \xi_\ell(s) = \frac{2\ell+1}{2} \int^{1}_{-1} d\mu \, \xi(s,\mu) L_\ell(\mu) ,
  ```

- Here, $\mu = \cos\theta$ is an even function, so the implementation

  ```math
  \xi_\ell(s) = (2\ell+1) \int^{1}_{0} d\mu \, \xi(s,\mu) L_\ell(\mu) .
  ```


## $P(k) \leftrightarrow \xi(r)$

-

- In general,

  ```math
  \begin{aligned}
  \xi(r) &= \frac{1}{(2\pi)^3} \int d^3k\, P(k)\, e^{i \mathbf{k}\cdot\mathbf{r}} , \\
  P(k) &= \int d^3r\, \xi(r)\, e^{-i \mathbf{k}\cdot\mathbf{r}} .
  \end{aligned}
  ```

- Under the assumption of isotropy, the 3D Fourier transform simplifies in spherical coordinates and takes the form of a Hankel transform. Specifically, the $\xi(r)$ and the $P(k)$ are related by the following equations:

  ```math
  \begin{aligned}
  \xi(r) &= \frac{1}{(2\pi)^3} \times \int_0^\infty dk\, k^2\, P(k)\, [4\pi  j_0(kr)] \\
  &= \frac{1}{2\pi^2} \int_0^\infty dk\, k^2\, P(k)\, j_0(kr).
  \end{aligned}
  ```

  ```math
  \begin{aligned}
  P(k) &= \int_0^\infty dr\, r^2\, \xi(r)\, [4\pi j_0(kr)] \\
  &= 4\pi \int_0^\infty dr\, r^2\, \xi(r)\, j_0(kr).
  \end{aligned}
  ```

  where \( j_0(kr) = \frac{\sin(kr)}{kr} \) is the spherical Bessel function of order zero.

-

## library memo

- C++ STL Legendre polynomials
  - https://en.cppreference.com/w/cpp/numeric/special_functions/legendre
  - https://en.cppreference.com/w/cpp/numeric/special_functions/assoc_legendre
- FFTlog
  - https://jila.colorado.edu/~ajsh/FFTLog/
