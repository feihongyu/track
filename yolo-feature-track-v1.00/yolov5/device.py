import torch
import pynvml

pynvml.nvmlInit()
choose = 0
temp = 0
if torch.cuda.is_available():
    for i in range(pynvml.nvmlDeviceGetCount()):
        if i == 1:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 0表示显卡标号
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.total/1024**2) #总的显存大小
        # print(meminfo.used/1024**2)  #已用显存大小
        # print(meminfo.free / 1024 ** 2)  # 剩余显存大小
        if meminfo.free > temp:
            temp = meminfo.free
            choose = i
    device = torch.device("cuda:" + str(choose))
else:
    device = torch.device('cpu')

# print("choose " + "cuda:" + str(choose))
