#include <Kokkos_Core.hpp>
#include <iostream>

template <typename T1, typename T2>
void axpy(T1 x, T1 y, T2 a, size_t N) 
{
  for (size_t i = 0; i < N; i++) 
  {
    y[i] += a * x[i]; 
  }
}

template <typename T>
void init(T x, T y, size_t N) 
{
  for (size_t i = 0; i < N; i++) 
  {
    x[i] = (i + 1) * 2.4;
    y[i] = (i + 1) * -1.2;
  }
}

template <typename T1, typename T2>
void kokkos_axpy(T1 x, T1 y, T2 a, size_t N) 
{
  Kokkos::parallel_for("axpy", N, KOKKOS_LAMBDA(const size_t i) {
    y[i] += a * x[i];
  });
}

template <typename T>
void kokkos_init(T x, T y, size_t N) 
{
  Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const size_t i) {
    x[i] = (i + 1) * 2.4;
    y[i] = (i + 1) * -1.2;
  });
}

template <typename T1, typename T2>
struct AxpyFunctor 
{
  T1 x;
  T1 y;
  T2 a;

  AxpyFunctor(T1 x_, T1 y_, T2 a_) : x(x_), y(y_), a(a_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const 
  {
    y[i] += a * x[i];
  }
};

template <typename T>
struct InitFunctor 
{
  T x;
  T y;

  InitFunctor(T x_, T y_) : x(x_), y(y_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const 
  {
    x[i] = (i + 1) * 2.4;
    y[i] = (i + 1) * -1.2;
  }
};

int main(int argc, char** argv)
{

  Kokkos::initialize(argc, argv);

  using fp_type = double;

  const fp_type a = 0.5;
  constexpr size_t N = 100;

  fp_type *x = (fp_type *) malloc(N * sizeof(fp_type));
  fp_type *y = (fp_type *) malloc(N * sizeof(fp_type));

  // option 1: direct function call
  // init(x, y, N);

  // option 2: lambda
  //kokkos_init(x, y, N);
  
  // option 3: functor
  Kokkos::parallel_for("init", N, InitFunctor(x, y));
  Kokkos::fence();
  
  std::cout << "First and last elements before axpy: " << std::endl
            << "x: " << x[0] << "," << x[N-1] << std::endl
            << "y: " << y[0] << "," << y[N-1] << std::endl;

  // option 1: direct function call
  // axpy(x, y, a, N);

  // option 2: lambda
  // kokkos_axpy(x, y, a, N);

  // option 3: functor
  Kokkos::parallel_for("axpy", N, AxpyFunctor(x, y, a));
  Kokkos::fence();

  // Check results
  std::cout << "First and last element (both should be zero):" << std::endl 
            << y[0] << "," << y[N-1] << std::endl;  

  Kokkos::finalize();
  return 0;

}
