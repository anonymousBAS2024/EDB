# Efficient Diffusion Bridge with Initial-value Correction Strategy for Super-Resolution

## Environment Requirements

 We run the code on a computer with `RTX-4090`, and `24G` memory. The code was tested with `python 3.9.0`, `pytorch 2.4.0`, `cudatoolkit 11.7.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```
# create a virtual environment
conda create --name EDB python=3.9.0

# activate environment
conda activate EDB

# install pytorch & cudatoolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## How to train
We train the model via running:

```
cd EDB/icme_code/config/sisr
python train_ours.py -opt=options/train/ours.yml
```
## How to test
```
cd EDB/icme_code/config/sisr
python test_ours.py -opt=options/test/ours.yml
```

