#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    int n = 20;
    int m = 20;

    Kokkos::View<int**> a("a", n, m); 

    // Subviews of boundaries
    auto top_boundary    = Kokkos::subview(a, 0, Kokkos::ALL());
    auto bottom_boundary = Kokkos::subview(a, n-1, Kokkos::ALL());
    auto left_boundary   = Kokkos::subview(a, Kokkos::ALL(), 0);
    auto right_boundary  = Kokkos::subview(a, Kokkos::ALL(), m-1);

    // Initialize boundaries in parallel
    // Top boundary
    Kokkos::parallel_for("InitTop", m, KOKKOS_LAMBDA(const int j) {
      top_boundary(j) = 1;
    });
    // Bottom boundary
    Kokkos::parallel_for("InitBottom", m, KOKKOS_LAMBDA(const int j) {
      bottom_boundary(j) = 2;
    });
    // Left boundary
    Kokkos::parallel_for("InitLeft", n, KOKKOS_LAMBDA(const int i) {
      left_boundary(i) = 3;
    });
    // Right boundary
    Kokkos::parallel_for("InitRight", n, KOKKOS_LAMBDA(const int i) {
      right_boundary(i) = 4;
    });

    // For copying to host, we need contiguous buffers in device
    Kokkos::View<int*> d_top("d_top", m);
    Kokkos::View<int*> d_bottom("d_bottom", m);
    Kokkos::View<int*> d_left("d_left", n);
    Kokkos::View<int*> d_right("d_right", n);

    // Copy subviews to contiguous device buffers
    Kokkos::deep_copy(d_top, top_boundary);
    Kokkos::deep_copy(d_bottom, bottom_boundary);
    Kokkos::deep_copy(d_left, left_boundary);
    Kokkos::deep_copy(d_right, right_boundary);

    // Create mirror views
    auto h_top    = Kokkos::create_mirror_view(d_top);
    auto h_bottom = Kokkos::create_mirror_view(d_bottom);
    auto h_left   = Kokkos::create_mirror_view(d_left);
    auto h_right  = Kokkos::create_mirror_view(d_right);

    // Copy to host
    Kokkos::deep_copy(h_top, d_top);
    Kokkos::deep_copy(h_bottom, d_bottom);
    Kokkos::deep_copy(h_left, d_left);
    Kokkos::deep_copy(h_right, d_right);

    Kokkos::fence();

    // Print out the boundary values
    std::cout << "Top Boundary: ";
    for(int j = 0; j < m; j++) {
      std::cout << h_top(j) << " ";
    }
    std::cout << std::endl;

    std::cout << "Bottom Boundary: ";
    for(int j = 0; j < m; j++) {
      std::cout << h_bottom(j) << " ";
    }
    std::cout << std::endl;

    std::cout << "Left Boundary: ";
    for(int i = 0; i < n; i++) {
      std::cout << h_left(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "Right Boundary: ";
    for(int i = 0; i < n; i++) {
      std::cout << h_right(i) << " ";
    }

  }
  Kokkos::finalize();
  return 0;
}
