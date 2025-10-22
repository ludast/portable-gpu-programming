#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q{default_selector_v, property::queue::in_order{}};
  std::cout << "Running on device: "
    << q.get_device().get_info<info::device::name>() << std::endl;

  // Initialize input and output memory on the host
  constexpr size_t N = 25'600;
  std::vector<int> x(N),y(N);
  int a=4;
  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 2);

  {
    // Create buffers for the host data or allocate memory usinggUSM
    // If USM + malloc_device() is used add the copy operations
    buffer<int> x_buf(x.data(), range<1>(N));
    buffer<int> y_buf(y.data(), range<1>(N));

    // Submit the kernel to the queue
    q.submit([&](handler& h) {
        // Create accessors if necessary
        auto x_acc = accessor(x_buf, h, read_only);
        auto y_acc = accessor(y_buf, h, read_write);

        h.parallel_for(
            //The kernel as a lambda
            range<1>(N), [=](id<1> i) {
            y_acc[i] = a * x_acc[i] + y_acc[i];
            }
            ); // end of parallel for
        });  // end of queue submission

    //DONE after the submission works
    //Checking the result inside the scope of the buffers using host_accessors
    {
      host_accessor res_y(y_buf, read_only);
      for (size_t i = 0; i < N; i++) {
        assert(res_y[i] == 6);
      }
    }
  }
  // If USM + malloc_device() is used add the copy operations 
  // TODO
  // Check that all outputs match expected value

  // If USM is used free the device memory
  // TODO
  // Check that all outputs match expected value
  bool passed = std::all_of(y.begin(), y.end(),
      [a](int val) { return val == a * 1 + 2; });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
    << std::endl;
  return (passed) ? 0 : 1;
}
