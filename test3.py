import os

import h5py
import numpy as np

from keras.preprocessing import image


class DataLoader(object):
    def __init__(self, data_folder, target_size=(214, 214)):
        self.data_folder = data_folder
        self.target_size = target_size
        self.x_data = []
        self.y_data = []
        self.labels = {}

    def make_h5_file(self, file_path, x, y):
        with h5py.File(file_path, "w") as f:
            f.clear()
            f.create_dataset(name="x_data", data=x)
            f.create_dataset(name="y_data", data=y)

    def make_h5(self, file_path, test_data_percent=0.):
        x_data, y_data = self.load_data()
        if test_data_percent > 0.5:
            raise Exception('test_data_percent can not bigger than 0.5')
        if test_data_percent > 0:
            all_len = len(x_data)
            test_len = np.ceil(all_len * test_data_percent)
            train_len = int(all_len - test_len)
            print("train data num: ", train_len, " test data num: ", test_len)
            prefix_name = os.path.splitext(file_path)[0]
            train_file_name = prefix_name + '_train.h5'
            x_train = x_data[:train_len]
            y_train = y_data[:train_len]
            self.make_h5_file(train_file_name, x_train, y_train)
            test_file_name = prefix_name + '_test.h5'
            x_test = x_data[train_len:]
            y_test = y_data[train_len:]
            self.make_h5_file(test_file_name, x_test, y_test)
        if test_data_percent == 0.:
            self.make_h5_file(file_path, x_data, y_data)

    def load_data(self, file_path=''):
        """
        获取培训数据
        :param file_path: 是否直接从已经生成好的数据文件中加载数据,
                          如果不设置，则扫描目录生成数据
        :return: (x_data, y_data)
        """
        if file_path == '':
            self._search_file()
            return self._shuffle(self.x_data, self.y_data)
            # return self.x_data, self.y_data
        else:
            data_file = h5py.File(file_path, "r")
            x = np.array(data_file["x_data"][:])
            y = np.array(data_file["y_data"][:])
            return x, y

    def load_labels_name(self, label_num):
        return self.labels[label_num]

    def _search_file(self):
        # 查找类型目录
        type_folder = os.listdir(self.data_folder)
        for i in range(len(type_folder)):
        # for i in range(0, 2):
            type_folder_path = os.path.join(self.data_folder, type_folder[i])
            if os.path.isdir(type_folder_path):
                print("search img in folder: ", type_folder_path)
                self.labels[i] = type_folder[i]
                self._ana_folder(type_folder_path, i)

    def _ana_folder(self, type_folder_path, type_folder_path_index):
        """
        加载图片至缓存中
        :param type_folder_path: 图片各类型目录路径
        :param type_folder_path_index: 图片各类型目录的顺序
        :return:
        """
        imgs = os.listdir(type_folder_path)
        for img in imgs:
            img_path = os.path.join(type_folder_path, img)
            if self._is_img(img_path):
                self._cache_data(self._reshape_img(img_path), type_folder_path_index)

    def _is_img(self, img_path):
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

    def _reshape_img(self, img_path):
        """
        图片reshape
        :param img_path: 目标图片
        :return:
        """
        reshape = image.load_img(img_path, target_size=self.target_size)
        img_data = image.img_to_array(reshape)
        return img_data

    def _cache_data(self, x_data, label):
        """
        缓存训练数据
        :param x_data: 单个图片
        :param y: 图片对应的类型标签，为图片所在目录的排序号
        :return:
        """
        self.x_data.append(x_data)
        self.y_data.append(label)

    def _shuffle(self, x, y):
        """
        顺序打乱
        :param x: 数据
        :param y: 标签
        :return: 打乱顺序后的数据和标签
        """
        index = np.arange(len(x))
        np.random.shuffle(index)
        x_tmp = np.array(x)
        del x
        y_tmp = np.array(y)
        del y
        x_shuffle = x_tmp[index]
        del x_tmp
        y_shuffle = y_tmp[index]
        del y_tmp
        return x_shuffle, y_shuffle


#
# loader = DataLoader('D:/retrain/retrain_data_color/backpack/train_data')
loader = DataLoader('D:/GitCode/python/keras_L/test')
# x, y = loader.load_data()
loader.make_h5('test/backpack_7_t.h5', test_data_percent=0.)
# x, y = loader.load_data(file_path='test/t1.h5')
# print(loader.load_labels_name(1))
#
#
# m = np.array([[1, 2, 4], [3, 4, 6], [5, 6, 7]])
# a = m[:2]
# b = m[2:]
# del m
# print(a)
# print(b)

# loader = DataLoader('D:/GitCode/python/keras_L/test/backpack_7_t.h5')
# x, y = loader.load_data('D:/GitCode/python/keras_L/test/backpack_7_t.h5')
# print(y)
# for i in range(len(y)):
#     if(y[i] == 0):
#         print(i)
