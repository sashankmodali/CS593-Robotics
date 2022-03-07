PyTorch implementation of MPnet with a pretrained model for a 2D planner.
Code has been taken from GitHub (https://github.com/MiaoDragon/MPNet-hw) and lightly modified for the purposes of this assignment.

* mpnet_test.py:
    * for generating path plan using existing models.
    * model_path: folder of trained model (model name by default is mpnet_epoch_[number].pkl)
    * N: number of environment to test on
    * NP: number of paths to test on
    * s: start of environment index
    * sp: start of path index
    * data_path: the folder of the data
    * result_path: the folder where results are stored (each path is stored in a different folder of environment)

* visualizer.py
    * obs_file: path of the obstacle point cloud file
    * path_file: path to the stored planned path file

* mpnet_Train.py:
    * for training the model.

* To run: create results, models, and data folder, and put the data into data folder. Execute mpnet_test.py to generate plan.
* Tested with python3.5.
