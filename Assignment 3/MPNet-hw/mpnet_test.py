from __future__ import print_function
from Model.end2end_model import End2EndMPNet
import Model.model as model
import Model.AE.CAE as CAE_2d
import numpy as np
import argparse
import os
import torch
from plan_general import *
import plan_s2d  # planning function specific to s2d environment (e.g.: collision checker, normalization)
import data_loader_2d
from torch.autograd import Variable
import copy
import os
import random
from utility import *
import utility_s2d
import progressbar
from matplotlib import pyplot as plt
import subprocess

def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # setup evaluation function and load function
    if args.env_type == 's2d':
        total_input_size = 2800+4
        AE_input_size = 2800
        mlp_input_size = 28+4
        output_size = 2
        IsInCollision = plan_s2d.IsInCollision
        load_test_dataset = data_loader_2d.load_test_dataset
        normalize = utility_s2d.normalize
        unnormalize = utility_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP

    mpNet = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size, \
                output_size, CAE, MLP)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # load previously trained model if start epoch > 0
    model_path='mpnet_epoch_%d.pkl' %(args.epoch)
    if args.epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        if args.reproducible:
            # set seed from model file
            torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
            torch.manual_seed(torch_seed)
            np.random.seed(np_seed)
            random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
    if args.epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))


    # load test data
    print('loading...')
    test_data = load_test_dataset(N=args.N, NP=args.NP, s=args.s, sp=args.sp, folder=args.data_path,rand=(args.part!=""))
    obc, obs, paths, path_lengths, envs_indices, paths_indices = test_data

    normalize_func=lambda x: normalize(x, args.world_size)
    unnormalize_func=lambda x: unnormalize(x, args.world_size)

    # test on dataset
    test_suc_rate = 0.
    DEFAULT_STEP = 0.01
    # for statistics
    n_valid_total = 0
    n_successful_total = 0
    sum_time = 0.0
    sum_timesq = 0.0
    min_time = float('inf')
    max_time = -float('inf')

    runtimes =[]
    success_rates=[]
    envs = []

    for i in range(len(paths)):
        n_valid_cur = 0
        n_successful_cur = 0
        results_files=""
        for run in range(args.n_runs):

            widgets = [
                f'planning: env={args.s + envs_indices[i]}, path=',
                progressbar.Variable('path_number', format='{formatted_value}', width=1), ' ',
                progressbar.Bar(),
                ' (', progressbar.Percentage(), ' complete)',
                ' success rate = ', progressbar.Variable('success_rate', format='{formatted_value}', width=4, precision=3),
                ' planning time = ', progressbar.Variable('planning_time', format='{formatted_value}sec', width=4, precision=3),
            ]
            bar = progressbar.ProgressBar(widgets = widgets)

            # save paths to different files, indicated by i
            # feasible paths for each env
            for j in bar(range(len(paths[0]))):
                time0 = time.time()
                if path_lengths[i][j]<2:
                    # the data might have paths of length smaller than 2, which are invalid
                    continue

                found_path = False
                n_valid_cur += 1
                path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                        torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]
                step_sz = DEFAULT_STEP
                MAX_NEURAL_REPLAN = 11
                for t in range(MAX_NEURAL_REPLAN):
                    if args.dropout_disabled:
                        mpNet.eval()
                    else:
                        mpNet.train()
                    path = neural_plan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz)
                    if not args.lvc_disabled:
                        path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    if feasibility_check(path, obc[i], IsInCollision, step_sz=step_sz):
                        found_path = True
                        n_successful_cur += 1
                        break

                time1 = time.time() - time0
                sum_time += time1
                sum_timesq += time1 * time1
                min_time = min(min_time, time1)
                max_time = max(max_time, time1)

                # write the path
                if type(path[0]) is not np.ndarray:
                    # it is torch tensor, convert to numpy
                    path = [p.numpy() for p in path]
                path = np.array(path)
                path_file = args.result_path+args.part+"/"+'env_%d/' % (envs_indices[i]+args.s)
                if not os.path.exists(path_file):
                    # create directory if not exist
                    os.makedirs(path_file)

                if found_path:
                    filename = f'path_{paths_indices[j]+args.sp}-run{run+1}.txt'
                    if run==0:
                        results_files = results_files + args.data_path +"e{}/".format(envs_indices[i]+args.s) + "path{}.dat ".format(paths_indices[j]+args.sp)
                    results_files = results_files + args.result_path +args.part+"/"+"env_{}/".format(envs_indices[i]+args.s) + "path_{}-run{}.txt ".format(paths_indices[j]+args.sp,run+1)
                else:
                    filename = f'path_{paths_indices[j]+args.sp}-run{run+1}-fail.txt'
                np.savetxt(path_file + filename, path, fmt='%f')

                success_rate = n_successful_cur / n_valid_cur if n_valid_cur > 0 else float('nan')

                if found_path:
                    bar.update(path_number=paths_indices[j]+args.sp, success_rate=success_rate, planning_time=time1)


            n_valid_total += n_valid_cur
            n_successful_total += n_successful_cur
        if args.vis_dir!="":
                subprocess.call('python visualizer.py --data-path {} --env-id {} --vis-dir {} --blind --path-file '.format(args.data_path,envs_indices[i]+args.s,args.vis_dir+args.part+"/") + results_files, shell=True)            
        if n_valid_total == 0:
            success_rate = avg_time = stdev_time = float('nan')
        else:
            success_rate = n_successful_total / n_valid_total if n_valid_total > 0 else float('nan')
            avg_time = sum_time / n_valid_total
            stdev_time = np.sqrt((sum_timesq - sum_time * avg_time) / (n_valid_total - 1)) if n_valid_total > 1 else 0
        print(f'cumulative: success rate={success_rate:.2f}, runtime (min/avg/max/stdev) = {min_time:.2f}/{avg_time:.2f}/{max_time:.2f}/{stdev_time:.2f}s')
        success_rates.append(success_rate)
        runtimes.append([min_time,avg_time,max_time,stdev_time])
        envs.append(args.s + envs_indices[i])
    lines = ["For the following environments","", str(envs),"success_rates :", str(success_rates), "", "runtimes (min/avg/max/stdev) :" , str(np.round(runtimes,2)), "", "respectively."]
    with open(args.result_path+args.part+"/report.txt", 'w') as f:
        f.writelines("\n".join(lines))

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='./models/',help='folder of trained model')
parser.add_argument('--N', type=int, default=1, help='number of environments')
parser.add_argument('--NP', type=int, default=1, help='number of paths per environment')
parser.add_argument('--s', type=int, default=0, help='start of environment index')
parser.add_argument('--sp', type=int, default=4000, help='start of path index')
parser.add_argument('--n-runs', type=int, default=1, help='number of runs')

# Model parameters
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data-path', type=str, default='./data/', help='path to dataset')
parser.add_argument('--result-path', type=str, default='./results/', help='folder to save paths computed')
parser.add_argument('--epoch', type=int, default=100, help='epoch of trained model to use')
parser.add_argument('--env-type', type=str, default='s2d', help='s2d for simple 2d')
parser.add_argument('--world-size', nargs='+', type=float, default=20., help='boundary of world')
parser.add_argument('--reproducible', default=False, action='store_true', help='use seed bundled with trained model')
parser.add_argument('--lvc-disabled', default=False, action='store_true', help='disable lvc for testing')
parser.add_argument('--dropout-disabled', default=False, action='store_true', help='disable dropout for testing')
parser.add_argument('--vis-dir', type=str, default='', help='directory for saving visualization. If empty, not saved')
parser.add_argument('--part', type=str, default='', help='assignment part. Saves folder accordingly')

args = parser.parse_args()
print(args)
main(args)
