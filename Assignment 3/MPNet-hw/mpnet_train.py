from __future__ import print_function
from Model.end2end_model import End2EndMPNet
import Model.model as model
import Model.AE.CAE as CAE_2d
import numpy as np
import argparse
import os
import torch
import data_loader_2d
from torch.autograd import Variable
import copy
import os
import random
from utility import *
import utility_s2d
import progressbar

from tensorboardX import SummaryWriter

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
    # decide dataloader, MLP, AE based on env_type
    if args.env_type == 's2d':
        total_input_size = 2800+4
        AE_input_size = 2800
        mlp_input_size = 28+4
        output_size = 2
        load_train_dataset = data_loader_2d.load_train_dataset
        normalize = utility_s2d.normalize
        unnormalize = utility_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP

    mpNet = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size, \
                        output_size, CAE, MLP)
    # setup loss
    if args.env_type == 's2d':
        loss_f = mpNet.loss

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # load previously trained model if start epoch > 0
    model_path='mpnet_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        if args.opt == 'Adagrad':
            mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpNet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpNet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
        elif args.opt == 'ASGD':
            mpNet.set_opt(torch.optim.ASGD, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))

    # load train and test data
    print('loading...')
    obstacles, dataset, targets, env_indices = load_train_dataset(N=args.N, NP=args.NP, folder=args.data_path)
    val_size = 1000
    val_dataset = dataset[:-val_size]
    val_targets = targets[:-val_size]
    val_env_indices = env_indices[:-val_size]
    # Train the Models
    print('training...')
    writer_fname = '%s_%f_%s' % (args.env_type, args.learning_rate, args.opt)
    writer = SummaryWriter('./runs/'+writer_fname)
    record_loss = 0.
    record_i = 0
    val_record_loss = 0.
    val_record_i = 0
    for epoch in range(args.start_epoch+1,args.epochs+1):
        sum_train_loss = 0.0
        sum_val_loss = 0.0

        widgets = [f'epoch{epoch} ',
            progressbar.Bar(),
            ' (batch ', progressbar.SimpleProgress(), ') ',
            progressbar.ETA(),
            ' train/val loss ',
            progressbar.Variable('train_loss', format='{formatted_value}'),'/',
            progressbar.Variable('val_loss', format='{formatted_value}'),
        ]
        bar = progressbar.ProgressBar(widgets = widgets)
        for i in bar(range(0, len(dataset), args.batch_size)):
            # randomly pick 100 data
            #bi = np.concatenate( (obstacles[env_indices[i:i+100]], dataset[i:i+100]), axis=1).astype(np.float32)
            bi = np.array(dataset[i:i+args.batch_size]).astype(np.float32)
            bobs = np.array(obstacles[env_indices[i:i+args.batch_size]]).astype(np.float32)
            bt = targets[i:i+args.batch_size]
            bi = torch.FloatTensor(bi)
            bobs = torch.FloatTensor(bobs)
            bt = torch.FloatTensor(bt)
            bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
            mpNet.zero_grad()
            bi=to_var(bi)
            bobs=to_var(bobs)
            bt=to_var(bt)
            mpNet.step(bi, bobs, bt)
            loss = loss_f(mpNet(bi, bobs), bt).item()
            sum_train_loss += loss * args.batch_size
            record_loss += loss
            record_i += 1
            if record_i % 100 == 0:
                writer.add_scalar('train_loss', record_loss / 100, record_i)
                record_loss = 0.

            ######################################
            # loss on validation set
            idx = i % val_size
            bi = np.array(val_dataset[idx:idx+100]).astype(np.float32)
            bobs = np.array(obstacles[val_env_indices[idx:idx+100]]).astype(np.float32)
            bt = val_targets[idx:idx+100]
            bi = torch.FloatTensor(bi)
            bobs = torch.FloatTensor(bobs)
            bt = torch.FloatTensor(bt)
            bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
            mpNet.zero_grad()
            bi=to_var(bi)
            bobs=to_var(bobs)
            bt=to_var(bt)
            loss = loss_f(mpNet(bi, bobs), bt).item()
            sum_val_loss += loss * args.batch_size

            val_record_loss += loss
            val_record_i += 1
            if val_record_i % 100 == 0:
                writer.add_scalar('val_loss', val_record_loss / 100, val_record_i)
                val_record_loss = 0.
            bar.update(train_loss = sum_train_loss / (i + args.batch_size), val_loss = sum_val_loss / (i + args.batch_size))

        # Save the models every 50 epochs
        if epoch % 50 == 0:
            model_path='mpnet_epoch_%d.pkl' %(epoch)
            save_state(mpNet, torch_seed, np_seed, py_seed, os.path.join(args.model_path,model_path))
            # test
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

parser = argparse.ArgumentParser()

parser.add_argument('--model-path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--data-path', type=str, default='../data/simple/')
parser.add_argument('--N', type=int, default=100, help='number of environments')
parser.add_argument('--NP', type=int, default=4000, help='number of paths per environment')

parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--env-type', type=str, default='s2d', help='s2d for simple 2d')
parser.add_argument('--world-size', nargs='+', type=float, default=20., help='boundary of world')
parser.add_argument('--opt', type=str, default='Adagrad')
args = parser.parse_args()
print(args)
main(args)
