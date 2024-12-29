<div align="center">

# `evotree` : a python library of processing phylogenetic tree
**Hengchi Chen**

[**Installation**](#installation)
</div>

## Installation
The `fossiler` package can be readily installed via `PYPI`. An example command is given below.

```
virtualenv -p=python3 ENV (or python3/python -m venv ENV)
source ENV/bin/activate
pip install evotree
```

Note that if users want to get the latest update, it's suggested to install from the source because the update on `PYPI` will be later than here of source. To install from source, the following command can be used.

```
git clone https://github.com/heche-psb/evotree
cd evotree
virtualenv -p=python3 ENV (or python3 -m venv ENV)
source ENV/bin/activate
pip install -r requirements.txt
pip install .
```

If there is permission problem in the installation, please try the following command.

```
pip install -e .
```

