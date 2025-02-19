# GroupCDL
Code for the paper *GroupCDL: Interpretable Denoising and Compressed Sensing
MRI Via Learned Group-Sparsity and Circulant Attention*, in IEEE Transactions
on Computational Imaging February 2025, doi: 10.1109/TCI.2025.3539021. 

## Install
1. If you're on an HPC, set your julia depot path to somewhere you can install files, e.g. scratch.
Throw this in your `.bashrc` and source it
```bash
export JULIA_DEPOT_PATH="/scratch/$USER/.julia" 
export JULIAUP_DEPOT_PATH="/scratch/$USER/.julia" 
```

2. Install via [juliaup](https://github.com/JuliaLang/juliaup) manager:
```bash
curl -fsSL https://install.julialang.org | sh
```

3. Install project environment
In the GroupCDL directory, start a julia instance and instantiate the project environment,
```bash
julia --project -t auto
julia> using Pkg; Pkg.instantiate()
```

4. Multi-GPU (once project environment instantiated)
```bash
julia --project -t auto -e "import MPI; MPI.install_mpiexecjl()"
```

## Usage
The following asssumes you have a Julia REPL for the project.

### Train your own model
Edit the configuration files in `configs/` to choose a network architecture, training, logging, and dataset details.
Then, in the Julia REPL, run,
```julia
julia> net, ps, st, ot = main(; network_config="config/groupcdl.yaml", closure_config="config/synthawgn_closure.yaml", data_config="config/image_data.yaml", warmup=true, train=true, verbose=true)
```

To train with multiple GPUs, we use MPI and call our main script as follows,
```bash
mpiexecjl -n <num_gpus> --project=. julia --project -t <num_cpus> main.jl --seed <seed> --train --warmup --verbose --mpi --config <path/to/config.yaml>"
 ```

Run `julia --project main.jl --help` for additional details on training from the commandline.

### Eval pretrained models
```julia
julia> net, ps, st, ot = main(; config="trained_nets/GroupCDL-S25/config.yaml", eval=true, verbose=true)
```

Optionally, you can provide alternate config files for the data and closure, ex., 
```julia
julia> net, ps, st, ot = main(; config="path/to/pretained_model/config.yaml", eval=true, eval_closure_config="config/synthawgn_closure.yaml", eval_data_config="config/image_data.yaml", verbose=true)
```

## Citation
```
@ARTICLE{janjusevicGroupCDL2025,
  author={Janjušević, Nikola and Khalilian-Gourtani, Amirhossein and Flinker, Adeen and Feng, Li and Wang, Yao},
  journal={IEEE Transactions on Computational Imaging}, 
  title={GroupCDL: Interpretable Denoising and Compressed Sensing MRI Via Learned Group-Sparsity and Circulant Attention}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TCI.2025.3539021}
}
```
