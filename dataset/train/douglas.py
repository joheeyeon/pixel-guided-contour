import math, random
import numpy as np

class Douglas:
    def __init__(self, D=3, num_vertices=None):
        self.D = D
        self.num_vertices = num_vertices

    def sample(self, poly):
        mask = np.zeros((poly.shape[0],), dtype=int)
        distance = np.zeros((poly.shape[0],), dtype=int)
        # mask[0] = 1 #rev7
        # endPoint = poly[0: 1, :] + poly[-1:, :]
        # endPoint /= 2
        # poly_append = np.concatenate([poly, endPoint], axis=0)
        # self.compress(0, poly.shape[0], poly_append, mask)
        # start_ind = 0 #rev7
        start_ind = random.randint(0, poly.shape[0]-1)  # rev8
        # poly_append = poly #rev7
        poly_append = np.roll(poly, -start_ind, 0)  # roll(-start_ind), rev8
        d = np.sum(np.abs(poly_append[0, :] - poly_append), -1)
        max_idx = np.argmax(d)
        dmax = d[max_idx]
        distance[max_idx] = dmax

        self.compress(0, max_idx, poly_append, mask, distance)
        self.compress(max_idx, poly_append.shape[0]-1, poly_append, mask, distance)
        if self.num_vertices is not None:
            # d_inds = np.argpartition(distance, -self.num_vertices+1)[-self.num_vertices+1:] #rev6
            d_inds = np.argpartition(distance, -self.num_vertices)[-self.num_vertices:]  # rev7
            # f"distance : {distance} / distance[d_inds]: {distance[d_inds]}")
            mask[:] = 0
            # mask[0] = 1 #rev7
            mask[d_inds[distance[d_inds] > 0]] = 1
            # f"d_inds : {d_inds} / mask : {mask}")
        # f"douglas mask : {mask.sum()}")
        mask = np.roll(mask, start_ind, 0)  # rev8
        return mask

    def compress(self, idx1, idx2, poly, mask, distance):
        p1 = poly[idx1, :]
        p2 = poly[idx2, :]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        m = idx1
        n = idx2
        # ✅ 재귀 호출의 종료 조건이자 에러 방지 코드
        #    시작점(m)과 끝점(n) 사이에 점이 없는 모든 경우(n <= m + 1)를
        #    처리하도록 조건을 강화하여, 빈 시퀀스가 생성되는 것을 막습니다.
        if (n <= m + 1):
            return
        d = abs(A * poly[m + 1: n, 0] + B * poly[m + 1: n, 1] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2) + 1e-4)
        max_idx = np.argmax(d)
        dmax = d[max_idx]
        max_idx = max_idx + m + 1

        # if self.num_vertices is not None:
        #     if mask.sum() < self.num_vertices:
        #         mask[max_idx] = 1
        #         self.compress(idx1, max_idx, poly, mask)
        #         self.compress(max_idx, idx2, poly, mask)
        # elif self.D is not None:
        #     if dmax > self.D:
        #         mask[max_idx] = 1
        #         self.compress(idx1, max_idx, poly, mask)
        #         self.compress(max_idx, idx2, poly, mask)
        if dmax > self.D:
            mask[max_idx] = 1
            distance[max_idx] = dmax
            self.compress(idx1, max_idx, poly, mask, distance)
            self.compress(max_idx, idx2, poly, mask, distance)
