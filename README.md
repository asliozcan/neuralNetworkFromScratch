# neuralNetwork

Construct neural network model from scrach

## Getting Started

### Prerequisities

```
sudo apt install git \
            python-pip \
```

```
pip install pipenv
```

### Installing 

Clone repo

```
git clone  git@git.basestech.com:asli.ozcan/neural-network-from-scratch.git
```

Install dependencies
```
pipenv install
```

## Running The Tests


### Coding style tests

```
pylint  // linter
```
### Documentation

Install
```
pip install sphinx
```

For documentation please use following scripts
```
cd docs/
sphinx-apidoc -o . ..
make html
firefox _build/html/index.html
```
### Test Codes

Install
```
pip install -U pytest
```

For running your test codes use following scripts

```
pytest
```
Create your test
```python
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5
```


# Built With

### Contributing

### Versioning

We use SemVer for versioning. For the versions available, see the tags on this repository.

## Authors

* **Asli Ozcan**

## Licence

### Acknowledgements