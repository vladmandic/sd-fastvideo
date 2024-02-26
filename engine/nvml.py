nvml_initialized = False


def get_reason(val):
    throttle = {
        1: 'gpu idle',
        2: 'applications clocks setting',
        4: 'sw power cap',
        8: 'hw slowdown',
        16: 'sync boost',
        32: 'sw thermal slowdown',
        64: 'hw thermal slowdown',
        128: 'hw power brake slowdown',
        256: 'display clock setting',
    }
    reason = ', '.join([throttle[i] for i in throttle if i & val])
    return reason if len(reason) > 0 else 'ok'


def get():
    import pynvml as nv
    global nvml_initialized # pylint: disable=global-statement
    if not nvml_initialized:
        nvml_initialized = True
        nv.nvmlInit()
    devices = []
    for i in range(nv.nvmlDeviceGetCount()):
        dev = nv.nvmlDeviceGetHandleByIndex(i)
        device = {
            'name': nv.nvmlDeviceGetName(dev),
            'version': {
                'cuda': nv.nvmlSystemGetCudaDriverVersion(),
                'driver': nv.nvmlSystemGetDriverVersion(),
                'vbios': nv.nvmlDeviceGetVbiosVersion(dev),
                'rom': nv.nvmlDeviceGetInforomImageVersion(dev),
                'capabilities': nv.nvmlDeviceGetCudaComputeCapability(dev),
            },
            'pci': {
                'link': nv.nvmlDeviceGetCurrPcieLinkGeneration(dev),
                'width': nv.nvmlDeviceGetCurrPcieLinkWidth(dev),
                'busid': nv.nvmlDeviceGetPciInfo(dev).busId,
                'deviceid': nv.nvmlDeviceGetPciInfo(dev).pciDeviceId,
            },
            'memory': {
                'total': round(nv.nvmlDeviceGetMemoryInfo(dev).total/1024/1024, 2),
                'free': round(nv.nvmlDeviceGetMemoryInfo(dev).free/1024/1024,2),
                'used': round(nv.nvmlDeviceGetMemoryInfo(dev).used/1024/1024,2),
            },
            'clock': { # gpu, sm, memory
                'gpu': [nv.nvmlDeviceGetClockInfo(dev, 0), nv.nvmlDeviceGetMaxClockInfo(dev, 0)],
                'sm': [nv.nvmlDeviceGetClockInfo(dev, 1), nv.nvmlDeviceGetMaxClockInfo(dev, 1)],
                'memory': [nv.nvmlDeviceGetClockInfo(dev, 2), nv.nvmlDeviceGetMaxClockInfo(dev, 2)],
            },
            'load': {
                'gpu': round(nv.nvmlDeviceGetUtilizationRates(dev).gpu),
                'memory': round(nv.nvmlDeviceGetUtilizationRates(dev).memory),
                'temp': nv.nvmlDeviceGetTemperature(dev, 0),
                'fan': nv.nvmlDeviceGetFanSpeed(dev),
            },
            'power': [round(nv.nvmlDeviceGetPowerUsage(dev)/1000, 2), round(nv.nvmlDeviceGetEnforcedPowerLimit(dev)/1000, 2)],
            'state': get_reason(nv.nvmlDeviceGetCurrentClocksThrottleReasons(dev)),
        }
        devices.append(device)
    return devices
