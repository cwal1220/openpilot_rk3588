#include "selfdrive/pandad/panda_comms.h"

#include <algorithm>
#include <stdexcept>

#include "common/swaglog.h"

static int init_usb_ctx(libusb_context **context) {
  int err = libusb_init(context);
  if (err != 0) {
    LOGE("libusb initialization error");
    return err;
  }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(*context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(*context, 3);
#endif

  return err;
}

PandaUsbHandle::PandaUsbHandle(std::string serial) {
  libusb_device **dev_list = nullptr;
  ssize_t num_devices = 0;

  int err = init_usb_ctx(&ctx);
  if (err != 0) {
    goto fail;
  }

  num_devices = libusb_get_device_list(ctx, &dev_list);
  if (num_devices < 0) {
    goto fail;
  }

  for (ssize_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(dev_list[i], &desc);
    if (desc.idVendor != 0xbbaa || desc.idProduct != 0xddcc) {
      continue;
    }

    int ret = libusb_open(dev_list[i], &dev_handle);
    if (dev_handle == nullptr || ret < 0) {
      goto fail;
    }

    unsigned char desc_serial[26] = {};
    ret = libusb_get_string_descriptor_ascii(dev_handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
    if (ret < 0) {
      goto fail;
    }

    hw_serial = std::string((char *)desc_serial, ret).c_str();
    if (serial.empty() || serial == hw_serial) {
      break;
    }

    libusb_close(dev_handle);
    dev_handle = nullptr;
  }

  if (dev_handle == nullptr) {
    goto fail;
  }

  libusb_free_device_list(dev_list, 1);
  dev_list = nullptr;

  if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
    libusb_detach_kernel_driver(dev_handle, 0);
  }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0 && err != LIBUSB_ERROR_BUSY) {
    goto fail;
  }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) {
    goto fail;
  }

  return;

fail:
  if (dev_list != nullptr) {
    libusb_free_device_list(dev_list, 1);
  }
  cleanup();
  throw std::runtime_error("Error connecting to USB panda");
}

PandaUsbHandle::~PandaUsbHandle() {
  std::lock_guard lk(usb_lock);
  cleanup();
  connected = false;
}

void PandaUsbHandle::cleanup() {
  if (dev_handle != nullptr) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
    dev_handle = nullptr;
  }

  if (ctx != nullptr) {
    libusb_exit(ctx);
    ctx = nullptr;
  }
}

std::vector<std::string> PandaUsbHandle::list() {
  libusb_context *context = nullptr;
  libusb_device **dev_list = nullptr;
  std::vector<std::string> serials;

  int err = init_usb_ctx(&context);
  if (err != 0) {
    return serials;
  }

  ssize_t num_devices = libusb_get_device_list(context, &dev_list);
  if (num_devices < 0) {
    LOGE("libusb can't get device list");
    goto finish;
  }

  for (ssize_t i = 0; i < num_devices; ++i) {
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(dev_list[i], &desc);
    if (desc.idVendor != 0xbbaa || desc.idProduct != 0xddcc) {
      continue;
    }

    libusb_device_handle *handle = nullptr;
    int ret = libusb_open(dev_list[i], &handle);
    if (ret < 0) {
      goto finish;
    }

    unsigned char desc_serial[26] = {};
    ret = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
    libusb_close(handle);
    if (ret < 0) {
      goto finish;
    }

    serials.push_back(std::string((char *)desc_serial, ret).c_str());
  }

finish:
  if (dev_list != nullptr) {
    libusb_free_device_list(dev_list, 1);
  }
  if (context != nullptr) {
    libusb_exit(context);
  }
  return serials;
}

void PandaUsbHandle::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
}

int PandaUsbHandle::control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout) {
  const uint8_t request_type = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;
  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(usb_lock);
  int err = 0;
  do {
    err = libusb_control_transfer(dev_handle, request_type, request, param1, param2, nullptr, 0, timeout);
    if (err < 0) {
      handle_usb_issue(err, __func__);
    }
  } while (err < 0 && connected);

  return err;
}

int PandaUsbHandle::control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout) {
  const uint8_t request_type = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;
  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(usb_lock);
  int err = 0;
  do {
    err = libusb_control_transfer(dev_handle, request_type, request, param1, param2, data, length, timeout);
    if (err < 0) {
      handle_usb_issue(err, __func__);
    }
  } while (err < 0 && connected);

  return err;
}

int PandaUsbHandle::bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  if (!connected) {
    return 0;
  }

  std::lock_guard lk(usb_lock);
  int err = 0;
  int transferred = 0;
  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      LOGW("Transmit buffer full");
      break;
    }
    if (err != 0 || length != transferred) {
      handle_usb_issue(err, __func__);
    }
  } while (err != 0 && connected);

  return transferred;
}

int PandaUsbHandle::bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  if (!connected) {
    return 0;
  }

  std::lock_guard lk(usb_lock);
  int err = 0;
  int transferred = 0;
  do {
    err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);
    if (err == LIBUSB_ERROR_TIMEOUT) {
      break;
    }
    if (err == LIBUSB_ERROR_OVERFLOW) {
      comms_healthy = false;
      LOGE_100("overflow got 0x%x", transferred);
    } else if (err != 0) {
      handle_usb_issue(err, __func__);
    }
  } while (err != 0 && connected);

  return transferred;
}
