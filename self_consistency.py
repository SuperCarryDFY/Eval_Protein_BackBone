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
import esm
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run_folding(_folding_model, sequence, save_path):
    """Run ESMFold on sequence."""
    with torch.no_grad():
        output = _folding_model.infer_pdb(sequence)

    with open(save_path, "w") as f:
        f.write(output)
    return output


def run_self_consistency(
        _folding_model, 
        decoy_pdb_dir: str,
        reference_pdb_path: str, 
        _pmpnn_dir = 'ProteinMPNN', 
        seq_per_sample=8
        ):
    """Run self-consistency on design proteins against reference protein.
    
    Args:
        decoy_pdb_dir: directory where designed protein files are stored.
        reference_pdb_path: path to reference protein file
        motif_mask: Optional mask of which residues are the motif.

    Returns:
        Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
        Writes ESMFold outputs to decoy_pdb_dir/esmf
        Writes results in decoy_pdb_dir/sc_results.csv
    """

    # Run PorteinMPNN
    output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
    process = subprocess.Popen([
        'python',
        f'{_pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
        f'--input_path={decoy_pdb_dir}',
        f'--output_path={output_path}',
        '--ca_only'])
    _ = process.wait()

    num_tries = 0
    ret = -1
    pmpnn_args = [
        'python',
        f'{_pmpnn_dir}/protein_mpnn_run.py',
        '--out_folder',
        decoy_pdb_dir,
        '--jsonl_path',
        output_path,
        '--num_seq_per_target',
        str(seq_per_sample),
        '--sampling_temp',
        '0.1',
        '--seed',
        '38',
        '--batch_size',
        '1',
        '--ca_only'  ### only for ca in genie version.
    ]

    while ret < 0:
        try:
            process = subprocess.Popen(
                pmpnn_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            ret = process.wait()
        except Exception as e:
            num_tries += 1
            print(f'#### Failed ProteinMPNN. Attempt {num_tries}/5 ####')
            torch.cuda.empty_cache()
            if num_tries > 4:
                raise e
    mpnn_fasta_path = os.path.join(
        decoy_pdb_dir,
        'seqs',
        os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
    )
    print('#### Finish ProteinMPNN Generation. Start ESMFold Generation.####')
    # assert False
    # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
    mpnn_results = {
        'tm_score': [],
        'sample_path': [],
        'header': [],
        'sequence': [],
        'rmsd': [],
    }
    esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
    os.makedirs(esmf_dir, exist_ok=True)
    fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
    sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
    for i, (header, string) in tqdm(enumerate(fasta_seqs.items())):

        # Run ESMFold
        esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
        _ = run_folding(_folding_model, string, esmf_sample_path)
        esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
        sample_seq = du.aatype_to_seq(sample_feats['aatype'])

        # Calculate scTM of ESMFold outputs with reference protein
        _, tm_score = metrics.calc_tm_score(
            sample_feats['bb_positions'], esmf_feats['bb_positions'],
            sample_seq, sample_seq)
        rmsd = metrics.calc_aligned_rmsd(
            sample_feats['bb_positions'], esmf_feats['bb_positions'])

        mpnn_results['rmsd'].append(rmsd)
        mpnn_results['tm_score'].append(tm_score)
        mpnn_results['sample_path'].append(esmf_sample_path)
        mpnn_results['header'].append(header)
        mpnn_results['sequence'].append(string)

    # Save results to CSV
    csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
    mpnn_results = pd.DataFrame(mpnn_results)
    mpnn_results.to_csv(csv_path)

def main():
    pdbs_path = '../runs/genie-base/version_0/samples/epoch_49999_pdb'
    sc_output_dir = os.path.join(pdbs_path, '../self_consistency')
    
    # load esmfold
    print('#### loading esmfold ####')
    _folding_model = esm.pretrained.esmfold_v1().eval()
    _folding_model = _folding_model.to(device)

    os.makedirs(sc_output_dir, exist_ok=True)
    import shutil
    for pdb in os.listdir(pdbs_path):
        print(f'#### Evaluating {pdb} ####')
        pdb_path = os.path.join(pdbs_path, pdb)
        sc_output_dir_per = os.path.join(sc_output_dir, pdb[:-4])
        os.makedirs(sc_output_dir_per, exist_ok=True)

        shutil.copy(pdb_path, os.path.join(
                sc_output_dir_per, os.path.basename(pdb_path)))
        run_self_consistency(_folding_model, sc_output_dir_per, pdb_path)
        

if __name__ == '__main__':
    main()