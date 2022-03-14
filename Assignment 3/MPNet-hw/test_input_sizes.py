import data_loader_r3d

import data_loader_2d


obc, obs, paths, path_lengths, envs_indices, paths_indices = data_loader_2d.load_test_dataset(N=1, NP=1,folder="./data/dataset-s2d/")

print("2d dataset sizes...")


print(len(obc[0][0]), len(obs[0]))


obc, obs, paths, path_lengths, envs_indices, paths_indices = data_loader_r3d.load_test_dataset(N=2, NP=2,folder="./data/dataset-c3d/")


print("3d dataset sizes...")


print(len(obc[0][0]), len(obs[0]))




