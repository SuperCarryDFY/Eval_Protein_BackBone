This repo is used for evaluate the in silico generated protein backbone through ProteinMPNN and ESMFold. 

## Useage

### Install the enviroment

```shell
conda env create -f se3.yml
```

### Install [ProteinMPNN](https://github.com/dauparas/ProteinMPNN/tree/main)

```bash
git clone git@github.com:dauparas/ProteinMPNN.git
```

### Run translate_to_pdb.py

If your code's output is *.npy instead of *.pdb, you should run translate_to_pdb.py to obtain *.pdb first. 

We default all the *.npy is under the `../rootdir/model_name/version/samples/epoch`, then the *pdb will be stored under the `../rootdir/model_name/version/samples/epoch_pdb.`

```shell
python translate_to_pdb.py
```

Make sure you have set the correct configures before run the code. 

### Run translate_to_pdb.py

This code contains run ProteinMPNN and ESMFold to evaluate the generated backbone. There are one path in self_consistency.py that should be set. 

- pdbs_path: the path you store the *pdb. If you have run the translate_to_pdb.py, you can set it to be`../rootdir/model_name/version/samples/epoch_pdb`.

```shell
python translate_to_pdb.py
```

Then we can found result in `sc_result.csv`.

## Acknowledge 
This repo is mainly based on [Framediff](https://github.com/jasonkyuyim/se3_diffusion) and [Genie](https://github.com/aqlaboratory/genie).