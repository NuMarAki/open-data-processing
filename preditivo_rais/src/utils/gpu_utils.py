def check_gpu_availability():
    return False

def ensure_gpu_available(force: bool = False):
    if force:
        raise EnvironmentError("GPU support was removed from this build.")
    return False

def get_gpu_info():
    return None

def set_gpu_device(device_id):
    raise EnvironmentError("GPU support was removed from this build.")
