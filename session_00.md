---
layout: default
---

# Setup

In this tutorial, you should follow the steps to setup your machine for
this module. Check out the __Technical Tutorials and Resources__ for
further readings.

## Grouping

Each team will have either two or three students.

## GitHub

In this module, we manage all the resources and projects in an GitHub Organization:

---
<div>
<h3 align="center">
    <a href="https://github.com/PnS2019">https://github.com/PnS2020</a>
</h3>
</div>
---

Therefore, an GitHub account is essential. If you don't have one, please
go to https://github.com/ and register an new account. After registration,
please submit your username to the TA so we can add you into the organization.

To get additional paid features, you can get the GitHub Education Pack: https://education.github.com/pack

## Your Computer

This section demonstrates the recommended configurations and tested.
You do not have to follow this part if you don't feel necessary.

### Miniconda

1. Download Miniconda for your system.

2. Install Miniconda by following [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

    Follow the instructions and make sure Miniconda is added in your bash configuration file such as `.bashrc` or `.zshrc`.

3. Close the current terminal and open another one (so that the bash configuration is loaded again). Type `python`, you should see something similar to this:

    ```
    Python 3.6.7 |Anaconda, Inc.| (default, Oct 23 2018, 19:16:44) 
    [GCC 7.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> 
    ```

4. Create an new environment for this module in your terminal

    ```bash
    $ conda create -n pns2020 python=3.6
    ```

    To use this environment, type this in the terminal
    ```bash
    $ source activate pns2020
    ```

### Install packages

Miniconda offers a minimum setup for Python. You will need to install some extra
packages for running experiments in this module. Simply run these commands sequentially

```bash
$ pip install numpy
$ pip install scipy
$ pip install scikit-learn
$ pip install scikit-image
$ pip install opencv-contrib-python
$ pip install matplotlib
$ pip install h5py
$ pip install torch
$ pip install torchvision
```

### IDE -- Pycharm

[PyCharm](https://www.jetbrains.com/pycharm/) is a Python IDE that has a beautiful interface and integrates all the features you will need for developing a Python project.

## (Optional) Google Colaboratory

[Colaboratory](https://colab.research.google.com/) is a free Jupyter notebook
environment that runs entirely in the cloud. Thanks to Google, they released
computing resources such as GPU and TPU for beginners to study Deep Learning.

In this module, we use this tool for demonstration and accelerating
training. To run code in Colaboratory, you will need a valid Google account.
