import os

import h5py
import numpy as np
import csv

from keras.preprocessing import image


def find_data_by_batch(folder_path, batch_size, test_percent=0.):
    x, y = find_img(folder_path)
    total = len(x)
    print('total: ', total)
    x_np = np.array(x)
    y_np = np.array(y)
    if test_percent > 0.:

        test_len = np.ceil(total * test_percent)
        train_len = int(total - test_len)
        print("train data num: ", train_len, " test data num: ", test_len)
        x_train = x_np[:train_len]
        y_train = y_np[:train_len]
        x_test = x_np[train_len:]
        y_test = y_np[train_len:]
    else:
        x_train = x_np
        y_train = y_np
        x_test = []
        y_test = []

    batchs = _build_batch(len(x_train), batch_size)

    x_train_batch = []
    y_train_batch = []
    for batch in batchs:
        x_train_batch.append(x_train[batch[0]: batch[1]])
        y_train_batch.append(y_train[batch[0]: batch[1]])
    return (x_train_batch, y_train_batch), (x_test, y_test)


def _build_batch(total, batch_size):
    batchs = []
    if total % batch_size == 0:
        batch_num = total / batch_size
    else:
        batch_num = total // batch_size + 1
    batch_num = int(batch_num)
    for i in range(batch_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > total:
            end = total - 1
        batchs.append([start, end])
    return batchs


def make_data(folder_path, file):
    x_s, y_s = find_img(folder_path)
    _make_csv_file(file, x_s, y_s)


def load_csv_data(csv_file):
    csv_reader = csv.reader(open(csv_file, encoding='utf-8'))
    for row in csv_reader:
        print(row)


def load_data(folder_path, target_size=(214, 214), test_percent=0.):
    x_s, y_s = find_img(folder_path)
    x = []
    for img in x_s:
        print('reshape: ', img)
        img_data = reshape_img(img, target_size)
        x.append(img_data)
    if test_percent > 0:
        all_len = len(x_s)
        test_len = np.ceil(all_len * test_percent)
        train_len = int(all_len - test_len)
        print("train data num: ", train_len, " test data num: ", test_len)
        x_train = x[:train_len]
        y_train = y_s[:train_len]
        x_test = x[train_len:]
        y_test = y_s[train_len:]
        return (x_train, y_train), (x_test, y_test)
    else:
        return (x, y_s), ([], [])


def batch_img_reshape(img_path_list, target_size):
    img_reshape = []
    for img_path in img_path_list:
        data = reshape_img(img_path, target_size)
        img_reshape.append(data)
    return img_reshape


def reshape_img(img_path, target_size):
    """
    图片reshape
    :param img_path: 目标图片
    :return:
    """
    reshape = image.load_img(img_path, target_size=target_size)
    img_data = image.img_to_array(reshape)
    return img_data


def find_img(folder_path):
    folders, _ = _search_file(folder_path)
    i = 0
    x_data = []
    y_data = []
    for folder in folders:
        if i > 2:
            break
        x_type, y_type = _ana_folder(folder, i)
        for x_tmp in x_type:
            x_data.append(x_tmp)
        for y_tmp in y_type:
            y_data.append(y_tmp)
        i = i + 1
    x_s, y_s = _shuffle(x_data, y_data)
    return x_s, y_s


def _search_file(data_folder):
    # 查找类型目录
    type_folder = os.listdir(data_folder)
    folders_path = []
    folders_name = {}
    i = 0
    for folder_name in type_folder:
        type_folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(type_folder_path):
            print("search img in folder: ", type_folder_path)
            # 目录名称
            folders_name[i] = folder_name
            # 目录路径
            folders_path.append(type_folder_path)
            i = i + 1
    return folders_path, folders_name


def _ana_folder(type_folder_path, type_folder_path_index):
    """
    加载图片路径信息至缓存中
    :param type_folder_path: 图片各类型目录路径
    :param type_folder_path_index: 图片各类型目录的顺序
    :return:
    """
    x_data = []
    y_data = []
    imgs = os.listdir(type_folder_path)
    for img in imgs:
        img_path = os.path.join(type_folder_path, img)
        if _is_img(img_path):
            x_data.append(img_path)
            y_data.append(type_folder_path_index)
    return x_data, y_data


def _is_img(img_path):
    """
    判断指定路径的文件是不是图片，只识别.jpg .jpeg .png
    :param img_path: 目标文件路径
    :return: 是：True, 否：False
    """
    ext = os.path.splitext(img_path)[1]
    if ext == '.jpg' or ext == 'jpeg' or ext == 'png':
        return True
    else:
        return False


def _shuffle(x_data, y_data):
    """
    顺序打乱
    :param x_data: 数据
    :param y_data: 标签
    :return: 打乱顺序后的数据和标签
    """
    index = np.arange(len(x_data))
    np.random.shuffle(index)
    x_tmp = np.array(x_data)
    del x_data
    y_tmp = np.array(y_data)
    del y_data
    x_shuffle = x_tmp[index]
    del x_tmp
    y_shuffle = y_tmp[index]
    del y_tmp
    return x_shuffle, y_shuffle


def _make_csv_file(file_path, x, y):
    file_header = ["img", "label"]
    # 写入数据
    csvFile = open(file_path, "w", newline='')
    writer = csv.writer(csvFile, dialect='excel')
    writer.writerow(file_header)
    for i in range(len(x)):
        row = [x[i], y[i]]
        writer.writerow(row)
    csvFile.close()


def make_h5_file(file_path, x, y):
    with h5py.File(file_path, "w") as f:
        f.clear()
        f.create_dataset(name="x_data", data=x, compression='gzip', compression_opts=9)
        f.create_dataset(name="y_data", data=y, compression='gzip', compression_opts=9)


def load_h5_data(file_path):
    data_file = h5py.File(file_path, "r")
    x = np.array(data_file["x_data"][:])
    y = np.array(data_file["y_data"][:])
    return x, y

# make_data('D:/retrain/retrain_data_color/backpack/train_data', 'D:/GitCode/python/keras_L/test/path_label_test.csv')

# (x_train, y_train), (x_test, y_test) = load_data('D:/retrain/retrain_data_color/backpack/train_data')
# (x_train, y_train), (x_test, y_test) = load_data('D:/GitCode/python/keras_L/test')
# make_h5_file('test/data_train.h5', x_train, y_train)

# csvFile = open('test/data_train.csv', "w", newline='')
# writer = csv.writer(csvFile, dialect='excel')
# for i in range(len(x_train)):
#     row = [x_train[i], y_train[i]]
#     writer.writerow(row)
# csvFile.close()

# csvFile = open('test/data_test.csv', "w", newline='')
# writer = csv.writer(csvFile, dialect='excel')
# for i in range(len(x_test)):
#     row = [x_test[i], y_test[i]]
#     writer.writerow(row)
# csvFile.close()


# load_csv_data('D:/GitCode/python/keras_L/test/data_train.csv')
