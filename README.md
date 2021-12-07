# HEART DISEASE AI DATATHON 2021

# How to start


1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Install Pytorch](https://pytorch.org/get-started/locally/)
3. Install dependencies
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. Put <echocardiography>dataset into <data> directory

```bash
cd Pytorch-UNet
mv echocardiography ./data
```

1. run training

```bash
python train.py --amp --type=A4C
```

# Description


fine-tuned from implementation of [U-Net in Pytorch for Kaggle's Carvana Image Masking Challenge](https://github.com/milesial/Pytorch-UNet)

- 20 epochs
- Loss function: Dice loss function
- Image preprocessing
- Optimizer: RMSprop

# Usage


**Note : Use Python 3.6 or newer**

## Training

name <echocardiography> dataset should be placed in ./data directory

```bash
> cd data
> ls
echocardiography

> cd ..
> python train.py --amp --epochs=20 --type=A2C --load=./checkpoints_A2C/checkpoint_epoch5_2021-12-06_02:21:25_best.pth
```

```bash
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

## Evaluation( with test set)

`python test.py --input=(input file directory) --label=(label file directory) --model=(model parameter file) --output=(output directory)` 

### Required arguments

- input : input image file directory (ex: ./data/echocardiography/validation/A2C )
- label: label image file directory (ex: ./data/echocardiography/validation/A2C )
- model: model parameter file (.pth)
- output: output mask file (both .png and .npy files) directory (ex: ./result/result_A2C)

### Optional arguments

- no-save: whether to save output masks or not
- scale: scale factor for input images

```bash
> python test.py -h
usage: test.py [-h] [--model FILE] --input INPUT [INPUT ...] --label INPUT
               [INPUT ...] --output INPUT [INPUT ...] [--no-save]
               [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Directory of input images (.png files)
  --label INPUT [INPUT ...], -l INPUT [INPUT ...]
                        Directory of label images (.npy files)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Directory of output images that will be produced
                        (.png, .npy files)
  --no-save, -n         Do not save the output masks
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

For example,

`python test.py --input=./data/echocardiography/validation/A2C --label=./data/echocardiography/validation/A2C --model=./checkpoints_A2C/checkpoint_epoch5_2021-12-06_02:21:25_best.pth --output=./result/result_A2C`

![Untitled](HEART%20DISEASE%20AI%20DATATHON%202021%20b8524490576948ac8f4f3fbeed956133/Untitled.png)

`python test.py --input=./data/echocardiography/validation/A4C --label=./data/echocardiography/validation/A4C --model=./checkpoints_A4C/checkpoint_epoch7_2021-12-06_02:49:59_best.pth --output=./result/result_A4C`

![Untitled](HEART%20DISEASE%20AI%20DATATHON%202021%20b8524490576948ac8f4f3fbeed956133/Untitled%201.png)

## Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```bash
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

You can specify which model file to use with `--model MODEL.pth`.

For example,

```bash
python predict.py --model=./checkpoints_A2C/checkpoint_epoch5_2021-12-06_02:21:25_best.pth --input=./data/echocardiography/validation/A2C/0801.png --output=test.png
```

![0801.png](HEART%20DISEASE%20AI%20DATATHON%202021%20b8524490576948ac8f4f3fbeed956133/0801.png)

![test.png](HEART%20DISEASE%20AI%20DATATHON%202021%20b8524490576948ac8f4f3fbeed956133/test.png)

# Original architecture


[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![Untitled](HEART%20DISEASE%20AI%20DATATHON%202021%20b8524490576948ac8f4f3fbeed956133/Untitled%202.png)