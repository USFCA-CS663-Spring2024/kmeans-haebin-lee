import sys

import numpy as np
import random
import heapq

class cluster:
    def __init__(self, k=5, max_iterations=100):
        self._k = k
        self._max_iterations = max_iterations
        self._centroids = np.random.rand(k, 2)

    def closest(self, p):
        min_v = sys.maxsize
        min_i = 0
        for i, center in enumerate(self._centroids):
            d = self.distance(p, center)
            if min_v > d:
                min_v = d
                min_i = i
        return min_i

    def distance(self, x, y):
        return np.linalg.norm(x - y)

    def balance(self, X, hypotheses):

        for _ in range(5):
            desired_size = round(len(X) / self._k)
            eps = round(desired_size * 0.1)
            count = np.bincount(hypotheses, minlength=self._k)
            max_count = max(count)
            min_count = min(count)
            max_idx = np.argmax(count)
            min_idx = np.argmin(count)

            if max_count <= desired_size + eps or desired_size - eps <= min_count:
                break

            pq = []
            need_to_update = np.where(hypotheses != min_idx)[0]
            for i in need_to_update:
                dist = self.distance(X[i], self._centroids[min_idx])
                pq.append((dist, i))
            heapq.heapify(pq)

            while min_count < desired_size - eps and pq:
                _, idx = heapq.heappop(pq)
                hypotheses[idx] = min_idx
                min_count += 1

            for i in range(self._k):
                cluster_points = X[hypotheses == i]
                if len(cluster_points) > 0:
                    self._centroids[i] = np.mean(cluster_points, axis=0)
        return hypotheses

    def fit(self, X, balanced=False):
        # X: 2D array

        hypotheses = np.full(len(X), -1)
        for _ in range(self._max_iterations):
            prev_centroids = np.copy(self._centroids)

            # Assignment - index of closest centroid to x
            for i, x in enumerate(X):
                idx = self.closest(x)
                if hypotheses[i] != idx:
                    hypotheses[i] = idx

            # Update
            for i in range(self._k):
                cluster_points = X[hypotheses == i]
                if len(cluster_points) > 0:
                    self._centroids[i] = np.mean(cluster_points, axis=0)

            hypotheses = hypotheses.astype(int)

            if np.array_equal(prev_centroids, self._centroids):
                break
        if balanced:
            hypotheses = self.balance(X, hypotheses)
        # A: A list (of length n) of the cluster hypotheses, one for each instances
        # B: A list (of length at most k) containing lists (each of length d) of the cluster centroids' values.
        return hypotheses, self._centroids


# t = cluster(5)
# X = np.array([[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]])
# h, centroids = t.fit(X, True)
# print(h, centroids)
