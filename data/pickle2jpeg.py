import os
import pickle
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# 定义目录路径
output_dir = '/data_hdd/users/pengzelin/HY_medical_seg/G_MLLA/MLLA-master/imagenet_mini'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# 创建目录
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

create_dir(train_dir)
create_dir(val_dir)

# 加载PKL文件
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 保存图像
def save_image(image_array, file_path):
    # 确保图像数据类型为uint8
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    # 转换为PIL图像
    image = Image.fromarray(image_array)

    # 保存为JPEG文件
    image.save(file_path, 'JPEG')

# 处理数据并保存
def process_data(data, split_dir):
    image_data = data['image_data']
    class_dict = data['class_dict']
    
    for class_name, indices in class_dict.items():
        class_dir = os.path.join(split_dir, class_name)
        create_dir(class_dir)
        for idx in indices:
            img_array = image_data[idx]
            img_filename = f"img{idx + 1}.jpeg"
            img_path = os.path.join(class_dir, img_filename)
            save_image(img_array, img_path)


# 加载PKL文件
train_data = load_pkl('/data_hdd/users/pengzelin/HY_medical_seg/G_MLLA/MLLA-master/minimagenet/data_miniimagenet/miniimagenet/mini-imagenet-cache-train.pkl')  # 替换为你的训练数据PKL文件路径
val_data = load_pkl('/data_hdd/users/pengzelin/HY_medical_seg/G_MLLA/MLLA-master/minimagenet/data_miniimagenet/miniimagenet/mini-imagenet-cache-test.pkl')      # 替换为你的验证数据PKL文件路径

# 处理并保存数据
process_data(train_data, train_dir)
process_data(val_data, val_dir)

print("转换完成！")
