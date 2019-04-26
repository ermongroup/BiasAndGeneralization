import sys
import os
import time
from matplotlib import pyplot as plt
sys.path.append('..')
sys.path.append('../DotsAndPie/')
from DotsAndPie.dataset.dataset_pie import *


class CombinationDataset:
    def __init__(self, num_combi):
        reader = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs.txt'))
        self.params = []
        while True:
            line = reader.readline().split()
            if len(line) == 0:
                break
            if int(line[0]) == num_combi:
                assert len(line) == num_combi + 1
                self.params = line[1:]
        if not self.params:
            print("The number of combinations you entered does not exist")
            assert False
        self.dataset = PieDataset(params=self.params)
        self.data_dims = self.dataset.data_dims

        size_list = [1, 3, 5, 7, 9]
        locx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        locy_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        color_list = [1, 3, 5, 7, 9]

        self.ground_truth_hist = np.zeros((5, 9, 9, 5), dtype=np.float)
        for param in self.params:
            self.ground_truth_hist[size_list.index(int(param[1])),
                                   locx_list.index(int(param[2])),
                                   locy_list.index(int(param[3])),
                                   color_list.index(int(param[4]))] = 1.0

    def next_batch(self, batch_size):
        return self.dataset.next_batch(batch_size)

    def get_histogram(self, samples):
        sizes = np.expand_dims(PieDataset.eval_size(samples), axis=1)
        locations = PieDataset.eval_location(samples)
        proportions = np.expand_dims(PieDataset.eval_color_proportion(samples), axis=1)
        # print(sizes.shape, locations.shape, proportions.shape)
        features = np.concatenate([sizes, locations, proportions], axis=1)
        # print(features.shape)
        size_bin = np.array([0.0, 0.45, 0.55, 0.65, 0.75, 1.00])
        locx_bin = np.array([-1.0, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 1.0])
        locy_bin = locx_bin
        proportion_bin = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        hist, _ = np.histogramdd(sample=features, bins=[size_bin, locx_bin, locy_bin, proportion_bin])
        # print(hist.shape)
        hist = hist.astype(np.float)
        hist /= np.sum(hist)
        return hist

    def get_pr(self, samples):
        precision_list = []
        recall_list = []
        hist = self.get_histogram(samples)
        hist_sort = np.flip(np.sort(hist.flatten()), 0)
        i = 0
        while True:
            threshold = hist_sort[i]
            bin_hist = (hist >= threshold).astype(np.float32)
            precision_list.append(np.sum(bin_hist * self.ground_truth_hist) / np.sum(self.ground_truth_hist))
            recall_list.append(np.sum(bin_hist * self.ground_truth_hist) / np.sum(bin_hist))
            if i < 10:
                i += 1
            elif i < 40:
                i += 2
            elif i < 100:
                i += 5
            elif i < 1000:
                i += 10
            else:
                break
        return np.array(precision_list), np.array(recall_list)

    def get_recall_at_precision(self, samples, precision=0.9):
        precision_list, recall_list = self.get_pr(samples)
        return np.max(recall_list[np.argwhere((precision_list - precision) >= 0)[0]])
