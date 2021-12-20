import numpy as np
import glob
import tqdm
import cv2
import os
import math
import datetime
import seaborn as sns
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


class DatasetRebalancing():
    def __init__(self):
        self.grid_info = {
            "0": [0, 0.33, 0, 0.33], "1": [0.33, 0.67, 0, 0.33], "2": [0.67, 1.00, 0, 0.33],
            "3": [0, 0.33, 0.33, 0.67], "4": [0.33, 0.67, 0.33, 0.67], "5": [0.67, 1.00, 0.33, 0.67],
            "6": [0, 0.33, 0.67, 1.00], "7": [0.33, 0.67, 0.67, 1.00], "8": [0.67, 1.00, 0.67, 1.00],
        }
        self.vector_list = None
        self.clustering_vector_list = None
        self.candidate_list = None
        self.train_list = None
        self.valid_list = None

    def color_extractor(self, img, topk=1):
        kmeans = KMeans(n_clusters=topk)
        kmeans = kmeans.fit(img.reshape((img.shape[1] * img.shape[0], 3)))
        color = [int(kmeans.cluster_centers_[0][0]), int(kmeans.cluster_centers_[0][1]),
                 int(kmeans.cluster_centers_[0][2])]
        return color

    def img2vector(self, img, grid_info):
        vector = []
        for grid in grid_info.values():
            st_x = int(grid[0] * img.shape[1])
            end_x = int(grid[1] * img.shape[1])
            st_y = int(grid[2] * img.shape[0])
            end_y = int(grid[3] * img.shape[0])
            crop = img[st_y:end_y, st_x:end_x]
            vector.extend(self.color_extractor(crop))
        return np.array(vector)

    def save_vector_info(self, vector, file_path):
        with open(file_path, 'a') as file:
            for index in range(len(vector)):
                item = vector[index]
                file.write(str(item))
                if (index != len(vector) - 1):
                    file.write(",")
            file.write("\n")

    def save_dataset(self, data, file_path):
        with open(file_path, 'w') as file:
            for index in range(len(data)):
                jpg_path = data[index]
                if index != (len(data) - 1):
                    file.write(jpg_path + "\n")
                else:
                    file.write(jpg_path)
        print(file_path + " is saved")

    def read_vector_info(self, file_path):
        vector_list = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            print("== Start load vector ==")
            print("load file path ->", file_path)
            for line in tqdm.tqdm(lines):
                line = line.replace("\n", "")
                vector = []
                for value in line.split(","):
                    vector.append(int(value))
                vector_list.append(vector)
            self.vector_list = vector_list
            print("== Finish load vector ==")

    def read_clustering_vector_info(self, file_path):
        vector_list = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            print("== Start load clustering vector ==")
            print("load file path :", file_path)
            for line in tqdm.tqdm(lines):
                line = line.replace("\n", "")
                vector = []
                for value in line.split(","):
                    vector.append(int(value))
                vector_list.append(vector)
            self.clustering_vector_list = np.array(vector_list)
            print("== Finish load clustering vector ==")

    def read_dataset(self, file_path):
        self.candidate_list = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            print("== Start load dataset ==")
            print("target :", file_path)
            for line in tqdm.tqdm(lines):
                cls, img_path = line.split(',')
                img_path = img_path.replace("\n", "")
                cls = int(cls)
                if cls in self.candidate_list:
                    lst = self.candidate_list[cls]
                    lst.append(img_path)
                    self.candidate_list[cls] = lst
                else:
                    self.candidate_list[cls] = [img_path]
            print("== Finish load dataset ==")

    def vector_clustering(self, result_path, topk=1000, verbose=1):
        vector_list = np.array(self.vector_list)
        kmeans = KMeans(n_clusters=topk, verbose=verbose)
        kmeans = kmeans.fit(vector_list)
        result = []
        for index in range(topk):
            result.append(kmeans.cluster_centers_[index])

        with open(result_path, 'w') as file:
            for vector in result:
                for index in range(len(vector)):
                    value = str(int(vector[index]))
                    file.write(value)
                    if index != len(vector) - 1:
                        file.write(",")
                file.write("\n")
        return result

    def vector_extraction(self, dataset_path):
        vector_list = []
        now = datetime.datetime.now()
        vector_filepath = now.strftime('%Y-%m-%d %H-%M-%S' + "_vector_info.txt")
        for img_path in tqdm.tqdm(dataset_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            vector = self.img2vector(img, self.grid_info)
            self.save_vector_info(vector, vector_filepath)
            vector_list.append(vector)
        self.vector_list = vector_list
        return vector_list

    def mse(self, vector1, vector2):
        error = np.mean(np.power(vector1 - vector2, 2), axis=1)
        return error

    def classification(self, result_path, dataset):
        self.candidate_list = {}
        if os.path.isfile(result_path):
            with open(result_path, 'w') as file:
                pass
        for img_path in tqdm.tqdm(dataset):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            vector = np.array(self.img2vector(img, self.grid_info))
            cur_vector_tile = np.tile(vector, (len(self.clustering_vector_list), 1))
            error_list = self.mse(cur_vector_tile, self.clustering_vector_list)
            cls = np.argmin(error_list)
            with open(result_path, 'a') as file:
                file.write(str(cls) + "," + img_path + "\n")
            if cls in self.candidate_list:
                lst = self.candidate_list[cls]
                lst.append(img_path)
                self.candidate_list[cls] = lst
            else:
                self.candidate_list[cls] = [img_path]

    def collection(self, train_lst_path="train.txt", valid_lst_path="valid.txt", valid_ratio=0.2, select_num=2,
                   verbose=True):
        '''
        :param train_lst_path:  train file path
        :param valid_lst_path:  valid file path
        :param valid_ratio: validation ratio
        :param select_option: data select num
        :param verbose:     plot show option
        :return:
        '''

        self.candidate_list = sorted(self.candidate_list.items())
        self.train_list = []
        self.valid_list = []
        cls_list = []
        count_list = []
        for dict in tqdm.tqdm(self.candidate_list):
            cls = dict[0]
            count = len(dict[1])
            cls_list.append(cls)
            count_list.append(count)
        count_list = np.array(count_list)
        min_cls = np.argmin(count_list)
        max_cls = np.argmax(count_list)
        min_count = count_list[min_cls]
        max_count = count_list[max_cls]
        mean_count = int(np.mean(count_list))
        if verbose:
            _fig = plt.figure(figsize=(20, 5))
            colors = sns.color_palette('hls', len(cls_list))
            plt.bar(cls_list, count_list, color=colors)
            plt.xlabel("class")
            plt.ylabel("count")
            plt.show()
            print("minimum_count = ", min_count)
            print("maximum_count = ", max_count)
            print("mean_count = ", mean_count)
        train_thresh_count = select_num
        for dict in tqdm.tqdm(self.candidate_list):
            file_list = dict[1]
            loop = len(file_list)
            if loop > train_thresh_count:
                loop = train_thresh_count
            for index in range(loop):
                self.train_list.append(file_list[index])

        valid_thresh_count = int(len(self.train_list) * valid_ratio)
        for dict in tqdm.tqdm(self.candidate_list):
            file_list = dict[1]
            loop = len(file_list)
            if loop == train_thresh_count:
                continue
            for index, file_path in enumerate(file_list[train_thresh_count:]):
                self.valid_list.append(file_path)
        random.shuffle(self.valid_list)
        self.valid_list = self.valid_list[:valid_thresh_count]
        self.save_dataset(self.train_list, train_lst_path)
        self.save_dataset(self.valid_list, valid_lst_path)


if __name__ == "__main__":
    target_dataset = glob.glob("/home/fsai2/sangmin/ObjectDetection/dteg/dataset/negative/**/*.jpg", recursive=True)
    dr = DatasetRebalancing()
    clustering_data = dr.vector_extraction(target_dataset)
    # dr.read_vector_info("./2021-10-25 13-01-51_vector_info.txt")
    vector_list = dr.vector_clustering("./negative_kmeans_vector_info.txt", topk=5000, verbose=1)
    dr.read_clustering_vector_info("./negative_kmeans_vector_info.txt")
    dr.classification("./negative_dataset_list.txt", target_dataset)
    dr.read_dataset("./negative_dataset_list.txt")
    dr.collection("negative_train.txt", "negative_valid.txt", select_num=2, verbose=False)
