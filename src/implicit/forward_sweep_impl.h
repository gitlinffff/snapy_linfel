#pragma once

// C/C++
#include <cmath>

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/math/ludcmp.h>
#include <snap/math/luminv.h>
#include <snap/snap.h>

#include "vic_solve_failure.h"

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

template <typename T, int N>
bool DISPATCH_MACRO ForwardSweep(Eigen::Matrix<T, N, N>* a,
                                 Eigen::Matrix<T, N, N>* b,
                                 Eigen::Matrix<T, N, N>* c,
                                 Eigen::Matrix<T, N, 1>* delta, T* du,
                                 double dt, int il, int iu, int dir, int ny,
                                 int stride1, int stride2, bool first_block,
                                 bool last_block) {
  Eigen::Matrix<T, N, 1> rhs;

  if constexpr (N == 3) {  // partial matrix
    rhs(0) = DU(IDN, il);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, il);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, il) / dt;
    rhs(2) = DU(IPR, il) / dt;
  } else {  // full matrix
    rhs(0) = DU(IDN, il);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, il);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, il) / dt;
    rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, il) / dt;
    rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, il) / dt;
    rhs(4) = DU(IPR, il) / dt;
  }

  int indx[N];
  Eigen::Matrix<T, N, N, Eigen::RowMajor> A;
  Eigen::Matrix<T, N, N, Eigen::RowMajor> Y;  // inverse from the pivot-test LU
  Eigen::Matrix<T, N, N + 1, Eigen::RowMajor> solved;

  // if (!first_block) {
  // RecvBuffer(a[il - 1], delta[il - 1], bblock);
  // a[il] = (a[il] - b[il] * a[il - 1]).inverse().eval();
  // delta[il] = a[il] * (rhs - b[il] * delta[il - 1]);
  // a[il] *= c[il];
  //} else {
  if constexpr (N > 4) {
    A = a[il];
    for (int n = 0; n < N; ++n) indx[n] = n;
    if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
    solved.template leftCols<N>() = c[il];
    solved.col(N) = rhs;
    lubksb(A, indx, solved);
    for (int row = 0; row < N; ++row)
      for (int col = 0; col < N + 1; ++col)
        if (!std::isfinite(solved(row, col)))
          return vic_fail_column(delta, il, iu);
    a[il] = solved.template leftCols<N>();
    delta[il] = solved.col(N);
  } else {  // retain the nonsingular small-matrix arithmetic
    A = a[il];
    if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
    luminv(A, indx, Y);  // invert from the pivot-test LU above
    a[il] = Y;
    delta[il] = a[il] * rhs;
    a[il] = a[il] * c[il];
  }
  // a[il] is finite: ludcmp refused nonfinite and near-singular matrices
  for (int row = 0; row < N; ++row)
    if (!std::isfinite(delta[il](row))) return vic_fail_column(delta, il, iu);
  //}

  for (int i = il + 1; i <= iu; ++i) {
    if constexpr (N == 3) {  // partial matrix
      rhs(0) = DU(IDN, i);
      for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
      rhs(0) /= dt;
      rhs(1) = DU(IVX + dir, i) / dt;
      rhs(2) = DU(IPR, i) / dt;
    } else {
      rhs(0) = DU(IDN, i);
      for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
      rhs(0) /= dt;
      rhs(1) = DU(IVX + dir, i) / dt;
      rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, i) / dt;
      rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, i) / dt;
      rhs(4) = DU(IPR, i) / dt;
    }

    a[i] -= b[i] * a[i - 1];

    if constexpr (N > 4) {
      A = a[i];
      for (int n = 0; n < N; ++n) indx[n] = n;
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      solved.template leftCols<N>() = c[i];
      solved.col(N) = rhs - b[i] * delta[i - 1];
      lubksb(A, indx, solved);
      for (int row = 0; row < N; ++row)
        for (int col = 0; col < N + 1; ++col)
          if (!std::isfinite(solved(row, col)))
            return vic_fail_column(delta, il, iu);
      a[i] = solved.template leftCols<N>();
      delta[i] = solved.col(N);
    } else {  // checked bypass path, inverting from its pivot-test LU
      A = a[i];
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      luminv(A, indx, Y);  // invert from the pivot-test LU above
      a[i] = Y;
      delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
      a[i] = a[i] * c[i];
    }
    // a[i] is finite: ludcmp refused nonfinite and near-singular matrices
    for (int row = 0; row < N; ++row)
      if (!std::isfinite(delta[i](row))) return vic_fail_column(delta, il, iu);
  }

  // SaveCoefficients(a, delta, il, iu);
  // if (!last_block) SendBuffer(a[iu], delta[iu], tblock);
  return true;
}

}  // namespace snap

#undef VOL
#undef DU
#undef W
