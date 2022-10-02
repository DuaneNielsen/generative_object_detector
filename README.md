#  Object detector from generative classification

### Install

Ubuntu 20.4

```
sudo apt-get install python3-tk
```

Create virtual environment

```
pip -m venv ./venv
./venv/bin/activate
```


Install pytorch for your CUDA version, eg CUDA 11.6

```commandline
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Install requirements

```commandline
pip install -r requirements.txt
```