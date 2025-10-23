#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>

// option 1: standard C++ version
template <typename T>
double dot_product(T x, T y, size_t N) 
{
  double result = 0.0;
  for (size_t i = 0; i < N; i++) 
  {
    result += x[i] * y[i];
  }
  return result;
}

template <typename T>
void init(T x, T y, size_t N) 
{
  for (size_t i = 0; i < N; i++) 
  {
    x[i] = cos(i * 1.2);
    y[i] = sin(i * -1.12);
  }
}

// option 2: Kokkos version with lambda
template <typename T>
double kokkos_dot_product(T x, T y, size_t N)
{
  double result = 0.0;
  Kokkos::parallel_reduce("dot_product", N,
    KOKKOS_LAMBDA(const size_t i, double& local_result) {
      local_result += x[i] * y[i];
    }, result);
  Kokkos::fence();
  return result;
}

template <typename T>
void kokkos_init(T x, T y, size_t N)
{
  Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(const size_t i) {
    x[i] = cos(i * 1.2);
    y[i] = sin(i * -1.12);
  });
  Kokkos::fence();
}

// option 3: Kokkos version with functor
template <typename T>
struct DotProductFunctor
{
  T x;
  T y;

  DotProductFunctor(T x_, T y_) : x(x_), y(y_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i, double& local_result) const
  {
    local_result += x[i] * y[i];
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
    x[i] = cos(i * 1.2);
    y[i] = sin(i * -1.12);
  }
};

int main(int argc, char** argv)
{

  Kokkos::initialize(argc, argv);

  using fp_type = double;

  constexpr size_t N = 100;

  fp_type *x = (fp_type *) malloc(N * sizeof(fp_type));
  fp_type *y = (fp_type *) malloc(N * sizeof(fp_type));

  // option 1: direct function call
  //init(x, y, N);

  // option 2: Kokkos lambda
  //kokkos_init(x, y, N);

  // option 3: Kokkos functor
  Kokkos::parallel_for("init", N, InitFunctor(x, y));
  Kokkos::fence();

  std::cout << "First and last elements before dot product: " << std::endl
            << "x: " << x[0] << "," << x[N-1] << std::endl
            << "y: " << y[0] << "," << y[N-1] << std::endl;

  // option 1: direct function call
  //auto result = dot_product(x, y, N);

  // option 2: Kokkos lambda
  //auto result = kokkos_dot_product(x, y, N);
  
  // option 3: Kokkos functor
  double result = 0.0;
  Kokkos::parallel_reduce("dot_product", N, DotProductFunctor(x, y), result);
  Kokkos::fence();

  // Check results
  std::cout << "Result (should be 6.78105): " << result << std::endl;

  Kokkos::finalize();
  return 0;
}
