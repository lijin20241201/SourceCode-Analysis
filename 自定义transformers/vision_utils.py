import os
import tensorflow as tf
import keras
import h5py  
import numpy as np
import matplotlib.pyplot as plt
def filter_imgs(f_pathlst):
    num_skipped = 0
    for path in f_pathlst:
        for name in os.listdir(path):
            fpath=os.path.join(path,name)
            try:
                fobj = open(fpath, "rb")#文件
                is_jfif = b"JFIF" in fobj.peek(10)#判断jfif是否在前10个字符里
                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)
            finally:
                fobj.close()
    return  num_skipped
def read_h5_data(path):
    # h5有自己的格式和内部结构，这些都需要由 h5py 库来解析，h5必须用r模式读取
    with h5py.File(path, 'r') as f: 
        images=f['images']
        labels=f['labels']
        images=np.array(images)
        labels=np.array(labels)
    return images,labels
def plot_history(history,item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
def use_gpu():
    gpus = tf.config.list_physical_devices('GPU')  
    if gpus:  
        # 如果有GPU，设置GPU资源使用率  
        try:  
            # 允许GPU内存按需增长  
            for gpu in gpus:  
                tf.config.experimental.set_memory_growth(gpu, True)  
            # 设置可见的GPU设备（这里实际上不需要，因为已经通过内存增长设置了每个GPU）  
            # tf.config.set_visible_devices(gpus, 'GPU')  
            print("GPU可用并已设置内存增长模式。")  
        except RuntimeError as e:  
            # 虚拟设备未就绪时可能无法设置GPU  
            print(f"设置GPU时发生错误: {e}")  
    else:  
        # 如果没有GPU  
        print("没有检测到GPU设备。")
