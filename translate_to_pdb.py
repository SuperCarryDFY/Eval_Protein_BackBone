import argparse
import numpy as np
import os
from analysis import utils as au
import icecream as ic
from analysis import metrics
import torch
import subprocess
from Pdata import utils as du
from biotite.sequence.io import fasta

def remove_pdb(outdir):
    file_list = os.listdir(outdir)
    file_list = [x for x in file_list if x[-3:] == 'pdb'] ### 
    for file in file_list:
        os.remove(os.path.join(outdir, file))
        print(f'#### successfully remove f{file} ####')


def main(args):
    print(args)
    rootdir = os.path.join('../', args.rootdir, args.model_name, 'version_{}'.format(args.model_version), 'samples')
    npydir = os.path.join(rootdir, 'epoch_{}'.format(args.model_epoch))
    outdir = os.path.join(rootdir, 'epoch_{}_pdb'.format(args.model_epoch)) if args.unwrite == False else os.path.join(outdir, 'for_test')
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)
    # remove_pdb(npydir)
    # assert False
    file_list = os.listdir(npydir)
    file_list = [x for x in file_list if x[-3:] != 'pdb'] ### 
    for file in file_list:
        ca = np.loadtxt(os.path.join(npydir,file), delimiter=',')
        protein_path = os.path.join(outdir, file).replace('npy', 'pdb')
        ### create other atom corr to be 0 
        length = ca.shape[0]
        N_atom = np.zeros((length, 1, 3))
        other_atom = np.zeros((length, 35, 3))
        ca = ca[:, np.newaxis, :]
        all_atom = np.concatenate([N_atom, ca, other_atom],axis=1)
        assert all_atom.shape == (length, 37, 3)

        sample_path = au.write_prot_to_pdb(
            all_atom,
            protein_path, 
            overwrite=True,
            no_indexing=True
        )
        print(f'#### translate {file} to pdb ####')


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    # parser.add_argument('-g', '--gpu', type=str, help='GPU device to use')
    parser.add_argument('-r', '--rootdir', type=str, help='Root directory (default to runs)', default='runs')
    parser.add_argument('-n', '--model_name', type=str, help='Name of Genie model', default='genie-base')
    parser.add_argument('-v', '--model_version', type=int, help='Version of Genie model', default='0')
    parser.add_argument('-e', '--model_epoch', type=int, help='Epoch Genie model checkpointed', default='49999')
    # for test use
    parser.add_argument('--unwrite', action='store_true')
    
    args = parser.parse_args()
    main(args)