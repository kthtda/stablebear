import stablebear as sb
from stablebear import gpu


def test_detect_nvidia_gpus_gives_a_dict_with_a_name_per_gpu():
    gpus = gpu.detect_nvidia_gpus()

    assert isinstance(gpus, list)
    assert all(isinstance(info, dict) and isinstance(info["name"], str) for info in gpus)


def test_nvidia_gpu_count_is_the_number_of_detected_gpus():
    assert gpu.nvidia_gpu_count() == len(gpu.detect_nvidia_gpus())


def test_has_nvidia_gpu_is_whether_any_gpu_is_detected():
    assert gpu.has_nvidia_gpu() == (len(gpu.detect_nvidia_gpus()) > 0)


def test_limit_gpus_accepts_the_number_of_gpus():
    sb.system.limit_gpus(gpu.nvidia_gpu_count())
