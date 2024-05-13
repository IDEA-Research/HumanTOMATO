import numpy as np

path = "/comp_robot/chenlinghao/OpenTMR/experiments/temos/debug--H3D-TMR-release-2/embeddings/val/epoch_99/motion_embedding.npy"

motion_embedding = np.load(path)
print(motion_embedding.shape)

# find the nearest neighbor of 0 index motion
distances = np.linalg.norm(motion_embedding - motion_embedding[0], axis=1)
print(distances, len(distances))

# find index and the distance of the nearest 4 neighbor
print(np.argsort(distances))
print(np.sort(distances))

# print(motion_embedding[3688])
# print(motion_embedding[0])
