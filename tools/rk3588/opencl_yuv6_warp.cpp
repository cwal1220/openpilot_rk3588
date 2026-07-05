#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *KERNEL_SOURCE = R"CLC(
__kernel void yuv6_warp(__global const uchar *src,
                        __global const float *m,
                        __global const float *uv_m,
                        __global uchar *dst,
                        int src_w,
                        int src_h,
                        int dst_w,
                        int dst_h) {
  int idx = get_global_id(0);
  int plane_w = dst_w / 2;
  int plane_h = dst_h / 2;
  int plane_size = plane_w * plane_h;
  int channel = idx / plane_size;
  int pixel = idx - channel * plane_size;
  int ox = pixel % plane_w;
  int oy = pixel / plane_w;

  if (channel < 4) {
    int dx = ox * 2 + ((channel == 2 || channel == 3) ? 1 : 0);
    int dy = oy * 2 + ((channel == 1 || channel == 3) ? 1 : 0);
    float sw = m[6] * dx + m[7] * dy + m[8];
    float sx_f = (m[0] * dx + m[1] * dy + m[2]) / sw;
    float sy_f = (m[3] * dx + m[4] * dy + m[5]) / sw;
    int sx = clamp((int)rint(sx_f), 0, src_w - 1);
    int sy = clamp((int)rint(sy_f), 0, src_h - 1);
    dst[idx] = src[sy * src_w + sx];
    return;
  }

  float sw = uv_m[6] * ox + uv_m[7] * oy + uv_m[8];
  float sx_f = (uv_m[0] * ox + uv_m[1] * oy + uv_m[2]) / sw;
  float sy_f = (uv_m[3] * ox + uv_m[4] * oy + uv_m[5]) / sw;
  int sx = clamp((int)rint(sx_f), 0, src_w / 2 - 1);
  int sy = clamp((int)rint(sy_f), 0, src_h / 2 - 1);
  int uv_offset = src_w * src_h + sy * src_w + sx * 2 + (channel == 5 ? 1 : 0);
  dst[idx] = src[uv_offset];
}
)CLC";

void set_error(char *error, std::size_t error_len, const std::string &message) {
  if (error == nullptr || error_len == 0) {
    return;
  }
  const std::size_t copy_len = std::min(error_len - 1, message.size());
  std::memcpy(error, message.data(), copy_len);
  error[copy_len] = '\0';
}

void check(cl_int status, const std::string &label) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(label + " failed: " + std::to_string(status));
  }
}

std::string device_name(cl_device_id device) {
  std::size_t size = 0;
  check(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size), "clGetDeviceInfo size");
  std::string value(size, '\0');
  check(clGetDeviceInfo(device, CL_DEVICE_NAME, size, value.data(), nullptr), "clGetDeviceInfo name");
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

std::vector<cl_platform_id> platforms() {
  cl_uint count = 0;
  check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs count");
  std::vector<cl_platform_id> values(count);
  check(clGetPlatformIDs(count, values.data(), nullptr), "clGetPlatformIDs values");
  return values;
}

std::vector<cl_device_id> gpu_devices(cl_platform_id platform) {
  cl_uint count = 0;
  const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
  if (status == CL_DEVICE_NOT_FOUND) {
    return {};
  }
  check(status, "clGetDeviceIDs count");
  std::vector<cl_device_id> values(count);
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, values.data(), nullptr), "clGetDeviceIDs values");
  return values;
}

cl_device_id choose_gpu() {
  std::vector<cl_device_id> fallback;
  for (cl_platform_id platform : platforms()) {
    for (cl_device_id device : gpu_devices(platform)) {
      if (fallback.empty()) {
        fallback.push_back(device);
      }
      if (device_name(device).find("Mali") != std::string::npos) {
        return device;
      }
    }
  }
  if (fallback.empty()) {
    throw std::runtime_error("no OpenCL GPU device found");
  }
  return fallback.front();
}

struct OpenClYuv6Warp {
  int src_w;
  int src_h;
  int dst_w;
  int dst_h;
  std::size_t src_size;
  std::size_t dst_size;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem src_buf = nullptr;
  cl_mem matrix_buf = nullptr;
  cl_mem uv_matrix_buf = nullptr;
  cl_mem dst_buf = nullptr;

  OpenClYuv6Warp(int src_width, int src_height, int dst_width, int dst_height)
      : src_w(src_width),
        src_h(src_height),
        dst_w(dst_width),
        dst_h(dst_height),
        src_size(static_cast<std::size_t>(src_width * src_height * 3 / 2)),
        dst_size(static_cast<std::size_t>(6 * (dst_width / 2) * (dst_height / 2))) {
    cl_int status = CL_SUCCESS;
    cl_device_id device = choose_gpu();
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &status);
    check(status, "clCreateCommandQueue");
    const char *sources[] = {KERNEL_SOURCE};
    const std::size_t lengths[] = {std::strlen(KERNEL_SOURCE)};
    program = clCreateProgramWithSource(context, 1, sources, lengths, &status);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (status != CL_SUCCESS) {
      std::size_t log_size = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      throw std::runtime_error("clBuildProgram failed: " + log);
    }
    kernel = clCreateKernel(program, "yuv6_warp", &status);
    check(status, "clCreateKernel");
    src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size, nullptr, &status);
    check(status, "clCreateBuffer src");
    matrix_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float), nullptr, &status);
    check(status, "clCreateBuffer matrix");
    uv_matrix_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float), nullptr, &status);
    check(status, "clCreateBuffer uv matrix");
    dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size, nullptr, &status);
    check(status, "clCreateBuffer dst");
  }

  ~OpenClYuv6Warp() {
    if (dst_buf != nullptr) clReleaseMemObject(dst_buf);
    if (uv_matrix_buf != nullptr) clReleaseMemObject(uv_matrix_buf);
    if (matrix_buf != nullptr) clReleaseMemObject(matrix_buf);
    if (src_buf != nullptr) clReleaseMemObject(src_buf);
    if (kernel != nullptr) clReleaseKernel(kernel);
    if (program != nullptr) clReleaseProgram(program);
    if (queue != nullptr) clReleaseCommandQueue(queue);
    if (context != nullptr) clReleaseContext(context);
  }

  void run(const std::uint8_t *src, const float *matrix, const float *uv_matrix, std::uint8_t *dst) {
    check(clEnqueueWriteBuffer(queue, src_buf, CL_FALSE, 0, src_size, src, 0, nullptr, nullptr), "write src");
    check(clEnqueueWriteBuffer(queue, matrix_buf, CL_FALSE, 0, 9 * sizeof(float), matrix, 0, nullptr, nullptr), "write matrix");
    check(clEnqueueWriteBuffer(queue, uv_matrix_buf, CL_FALSE, 0, 9 * sizeof(float), uv_matrix, 0, nullptr, nullptr), "write uv matrix");
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf), "arg src");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &matrix_buf), "arg matrix");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &uv_matrix_buf), "arg uv matrix");
    check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_buf), "arg dst");
    check(clSetKernelArg(kernel, 4, sizeof(int), &src_w), "arg src_w");
    check(clSetKernelArg(kernel, 5, sizeof(int), &src_h), "arg src_h");
    check(clSetKernelArg(kernel, 6, sizeof(int), &dst_w), "arg dst_w");
    check(clSetKernelArg(kernel, 7, sizeof(int), &dst_h), "arg dst_h");
    const std::size_t global = dst_size;
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "run kernel");
    check(clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, dst_size, dst, 0, nullptr, nullptr), "read dst");
  }
};

}

extern "C" void *opencl_yuv6_warp_create(int src_w, int src_h, int dst_w, int dst_h, char *error, std::size_t error_len) {
  try {
    return new OpenClYuv6Warp(src_w, src_h, dst_w, dst_h);
  } catch (const std::exception &exc) {
    set_error(error, error_len, exc.what());
    return nullptr;
  }
}

extern "C" int opencl_yuv6_warp_run(void *handle, const std::uint8_t *src, const float *matrix,
                                     const float *uv_matrix, std::uint8_t *dst, char *error,
                                     std::size_t error_len) {
  try {
    if (handle == nullptr) {
      throw std::runtime_error("OpenCL warp handle is null");
    }
    static_cast<OpenClYuv6Warp *>(handle)->run(src, matrix, uv_matrix, dst);
    return 0;
  } catch (const std::exception &exc) {
    set_error(error, error_len, exc.what());
    return -1;
  }
}

extern "C" void opencl_yuv6_warp_destroy(void *handle) {
  delete static_cast<OpenClYuv6Warp *>(handle);
}
