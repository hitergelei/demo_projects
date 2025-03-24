
import numpy as np
from typing import List, Tuple

class Boxes:
    def __init__(self, cutoff: float):
        self.box_size = 1.001 * cutoff
        self.boxes = {}

    def insert(self, i: int, x: float, y: float, z: float):
        key = (
            int(x // self.box_size),
            int(y // self.box_size),
            int(z // self.box_size),
        )
        self.boxes.setdefault(key, []).append(i)

    def iter(self):
        return self.boxes.items()

    def iter_neighbors(self, key):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    new_key = (key[0] + i, key[1] + j, key[2] + k)
                    if new_key in self.boxes:
                        neighbors.extend(self.boxes[new_key])
        return neighbors

def neighbor_list_ijdD(positions: List[List[float]], cutoff: float, self_interaction: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(positions)
    boxes = Boxes(cutoff)

    for i in range(n):
        x, y, z = positions[i][0], positions[i][1], positions[i][2]
        boxes.insert(i, x, y, z)

    src, dst, dist, rel = [], [], [], []

    for key, value in boxes.iter():
        for j in boxes.iter_neighbors(key):
            for i in value:
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dz = positions[j][2] - positions[i][2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < cutoff * cutoff:
                    if self_interaction or i != j:
                        src.append(i)
                        dst.append(j)
                        dist.append(np.sqrt(d2))
                        rel.append([dx, dy, dz])

    return np.array(src), np.array(dst), np.array(dist), np.array(rel)

def neighbor_list_ij(positions: List[List[float]], cutoff: float, self_interaction: bool) -> Tuple[np.ndarray, np.ndarray]:
    n = len(positions)
    boxes = Boxes(cutoff)

    for i in range(n):
        x, y, z = positions[i][0], positions[i][1], positions[i][2]
        boxes.insert(i, x, y, z)

    src, dst = [], []

    for key, value in boxes.iter():
        for j in boxes.iter_neighbors(key):
            for i in value:
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dz = positions[j][2] - positions[i][2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < cutoff * cutoff:
                    if self_interaction or i != j:
                        src.append(i)
                        dst.append(j)

    return np.array(src), np.array(dst)
