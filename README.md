# PolyACO+
An algorithm using Ant Colony Optimization with polygons and ray casting for classification.

## Installation
The easiest way to install the project is by using [Conda](http://conda.pydata.org/). Conda is a Python package manager and environment manager that makes it easy to set up and install Python environments. Follow the instructions on [Condas website](http://conda.pydata.org/docs/install/quick.html) to get started.

After installing Conda, `cd` into the project directory and install the project environment with:

```bash
$ conda env create -f environment.yml python=3.5
```

This will install a full conda environment named *acoc*. Activate the enviroment by running:

```
$ source activate acoc
```

## Configuration
The project looks for a configuration file called *config.py* in the root folder. This file is ignored by the VCS so that you can make changes without affecting the repository. To get started with a config file simply copy *config_template.py* and rename it to *config.py*. Then you can freely change the contents of the config file without affecting the VCS.

### CUDA
The algorithm is optimized for Nvidia GPUs and depends on CUDA. To run the project without CUDA, set `'gpu': false` in *config.py*.

## Usage
A complete classification example can be found in *demo.py* in the root folder of the repository. The demo use the classifier configuration specified in *config.py*. 
