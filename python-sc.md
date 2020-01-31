---
layout: default
---

# Setting up your laptop for Scientific Computing with Python

## General Advice on Operating System

Generally, you should use either a modern Linux or `*nix` distribution.
For most users, we recommend [Ubuntu](https://www.ubuntu.com/).
If you have a Mac, `macOS` is a custom flavor of `*nix` OS.
Please do not try to install other systems such as Ubuntu or Arch Linux
on a Mac if you don't know what you are doing.

You should avoid using Windows as much as possible.
For this module, we can only offer limited help if you are using Windows.

## Which Python Distribution Should I Use?

If you are a newbie to Python, you would go to the official Python website,
download the installation file and install it on your computer.
In fact, this may be the worst thing you can do.

So, in most modern Linux and `*nix` operating systems, the Python distribution
is installed for managing some system services by default.
The system-installed Python can be used by the user for sure.
However, because this distribution manages system services and needs constant
writing privileges to some directories, we strongly recommend that you do not
touch the system-installed Python as a new user.
Moreover, you could encounter big problems when you are trying to install packages that build from scratch.

Instead, what we need is a Python distribution that has its own environment
and can be easily removed when we want to.
So the answer is [Miniconda](https://conda.io/miniconda.html),
a Python distribution that is built for Scientific Computing.

### Miniconda setup instructions

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

## Which Python Version Should I Use?

Python 3.6

__Remark__: The support of Python 2.7 has ended. Do not use Python 2.7.

__Remark__: Do not use Python 3.7 for now as many softwares don't have support for this version yet.

## How to Install Other Python Packages?

Generally, with Miniconda, we have three ways of installing other Python packages.

1. Use `pip`. `pip` is the official Python packaging system that manages package installation, uninstallation, version control, custom package building, etc. You can install additional Python packages by

    ```bash
    $ pip install some-package-name
    ```

    If the package is available in [PyPi](https://pypi.org/), `pip` will automatically pull the software from the website and install it.

2. Use `conda`. Miniconda uses the packaging system `conda` to manage packages and libraries installation. At heart, `conda` does package, dependency and environment management for any language. `conda` can pull and install a pre-built package from a specific server and resolve the dependency accordingly.

    ```bash
    $ conda install some-package-name
    ```

3. Use `setup.py`. A decent Python library usually has a `setup.py` script. With this script, you can install the package via

    ```bash
    $ python setup.py install
    ```

## Do I Need Anything Else?

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
$ pip install tensorflow
```

You should familiarize yourself with the software packaging system on your computer.
For Debian family (including Ubuntu), that is `apt-get` or `apt`. For macOS, that is either `MacPorts` or `homebrew` (strongly recommended).

## IDE (Integrated Development Environment)

There are many different IDEs for Python on the market. Here, we list some of
them that generally deliver fantastic coding experience.

### Say NO to Jupyter Notebook

If you have taken a Python class or Python 101, you have probably seen that
instructors like Jupyter Notebook as a default presentation tool.
They even walk you through the steps so that you are comfortable with
this kind of coding style.

However, we are strongly against this paradigm of coding.
It is true that the Jupyter Notebook offers a nice presentation style.
But it is not designed for managing and engineering serious projects.

Therefore, we advise that you do not use Jupyter Notebook.

### What do you need from an IDE?

+ __Code auto-completion__: Let's face it, no one is gonna remember hundreds or even thousands of APIs.
+ __Code navigation__: If you want to check on some definitions or files, you need to get there fast.
+ __Running the code__: It should be easy to run the code.
+ __Debugging__: If there is anything wrong with the code, you should be able to find out which line is doing things wrongly.
+ __Source code version control__: Make sure your code is maintained and updated in a timely and efficient manner.
+ __Static code checker__: Modern software engineering promotes the idea of producing high-quality _human readable_ code. Over the years, people have transformed specific coding styles into a static code checker where the checker evaluates your code formatting and corrects obvious errors.

### PyCharm

[PyCharm](https://www.jetbrains.com/pycharm/) is a new Python IDE that has a beautiful interface and integrates all the features you will need for developing a Python project.

### Atom

[Atom](https://atom.io/) is a very popular text editor that is built and maintained mainly by GitHub. This editor is modern and naturally integrates some of the best software engineering practices. You can find tons of additional packages that help you configure the editor to a Python IDE.

### Eclipse+PyDev

If you are a [Eclipse](http://www.eclipse.org/) user, perhaps you would be happy to know that there is a dedicate Eclipse plugin for Python. [PyDev](http://www.pydev.org/) takes advantages of the powerful Eclipse compiling and debugging framework.

### Vim

For the experienced user, we recommend [Vim](https://www.vim.org/) or its community-driven fork [neovim](https://neovim.io/). By configuring Vim, you will be able to get a fast editor that has all the IDE features you want while being lightweight. Additionally, Vim is a very flexible editor which allows you to efficiently do more things compared to other editors.

## Further Readings

+ [NSC-GPU-GUIDE](https://github.com/duguyue100/NSC-GPU-GUIDE): this repository contains some setup scripts that we use at INI for setting up computing resources for master students.
