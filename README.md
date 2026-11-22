# BanglaTaka: Bangladeshi banknote classification with a CNN built from scratch

CVPR (CSC 4254) midterm assignment, Fall 26-27.

| | |
|---|---|
| **Student** | Tahmid Jawad Shafi_22-49127-3|
| **Section** | A |
| **Notebook (Kaggle)** | https://www.kaggle.com/code/tahmidxshafi/notebook5e5c919617 |
| **Dataset (Kaggle)** | https://www.kaggle.com/datasets/tahmidxshafi/banglataka-bangladeshi-banknote-images |

## What this project does

A small convolutional neural network, written in PyTorch without pretrained weights, classifies images of Bangladeshi
banknotes into 9 denominations. The notebook builds the network, trains it, and compares design choices one at a time
(number of conv layers, channels, kernel size, hidden layer, dropout, augmentation, padding, pooling, stride).
The goal of the assignment is to experiment and learn, not to reach the highest possible accuracy.

## Dataset

- **BanglaTaka**, 9 denominations: 2, 5, 10, 20, 50, 100, 200, 500, 1000 Taka.
- 5,073 images, 1.71 GB (the assignment limits were at least 2,000 images and at most 2.5 GB).
- Source: Mendeley Data, DOI [10.17632/3cv2sypkkh.1](https://doi.org/10.17632/3cv2sypkkh.1); described in Data in Brief,
  DOI [10.1016/j.dib.2025.111853](https://doi.org/10.1016/j.dib.2025.111853). Authors: Md. Naimul Islam Nuhash, Sadia Akter.
  Licence: CC BY 4.0. The images are not stored in this repository; they were re-uploaded unchanged to Kaggle for the course.

| Class | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| Images | 445 | 660 | 419 | 480 | 416 | 565 | 423 | 1243 | 422 |

The classes are imbalanced (largest / smallest = 3.0), so macro-F1 is reported next to accuracy.

## Method

- Images resized to 128 x 128. Stratified split 70/15/15: 3,551 train, 761 validation, 761 test.
- Augmentation (rotation up to 15 degrees, brightness / contrast / saturation jitter) is applied to training images only.
  No horizontal flip, because mirroring a banknote reverses its printed text.
- Adam optimiser, fixed learning rate 0.001, batch size 32, 20 epochs, cross-entropy loss. The weights of the epoch with
  the highest validation accuracy are kept.
- Baseline network: 3 conv blocks (16-32-64 channels, 3x3 kernels, padding 1, ReLU, 2x2 max pooling), then a hidden
  layer of 128 neurons and a 9-way output.
- Experiments were compared on the validation set; the test set was used once, for the final model.

## Results

Final model (E6: conv channels 32-64-128, no dropout, no augmentation; 4,288,841 trainable parameters):

| Metric | Value |
|---|---|
| Test accuracy | 98.69% (n = 761, approx. 95% interval +/- 0.8 points) |
| Test macro-F1 | 0.987 |
| Misclassified test images | 10 of 761 |
| Validation accuracy / loss at best epoch (20) | 99.21% / 0.0552 |

Validation accuracy of every run (one seed each):

| Run | Change from baseline | Val acc | Parameters |
|---|---|---|---|
| Round 1 baseline | none | 0.988 | 2,122,025 |
| Round 2 | augmentation + dropout 0.5 | 0.984 | 2,122,025 |
| E3 | dropout 0.5 only | 0.988 | 2,122,025 |
| E4 | augmentation only | 0.989 | 2,122,025 |
| E5 | channels 8-16-32 | 0.980 | 1,055,897 |
| E6 | channels 32-64-128 | 0.992 | 4,288,841 |
| E7 | 2 conv layers | 0.989 | 4,200,681 |
| E8 | 4 conv layers | 0.972 | 1,147,305 |
| E9 | 5x5 kernel | 0.989 | 2,163,753 |
| E10 | hidden layer 64 | 0.988 | 1,072,809 |
| E11 | hidden layer 256 | 0.987 | 4,220,457 |
| E12 | no padding | 0.987 | 1,630,505 |
| E13 | average pooling | 0.986 | 2,122,025 |
| E14 | stride 2 | 0.951 | 57,641 |

With 761 validation images, differences below about 0.8 points are within noise. Only E8 and E14 are clearly lower than
the baseline. Re-running the notebook gave slightly different validation numbers, so treat small differences as noise.

A small CNN trained from scratch classifies the nine denominations with about 98.7% test accuracy. Regularisation (dropout, augmentation) reduced the train/validation gap but did not raise validation accuracy, and changes in width, depth, kernel size, hidden size, padding and pooling mostly stayed within noise. Only the four-conv-layer model (E8) and the stride-2 model (E14) were clearly worse than the baseline.

## Limitations

- Every configuration was trained once with one seed.
- Photos of the same note taken in bursts may appear in both train and test, which can inflate accuracy. The dataset has
  no group labels, so a random split cannot rule this out.
- Images are stretched to a square, which distorts the notes slightly.
- A pretrained network would probably do better; it was not tested.

## Run it

1. Upload or attach the dataset on Kaggle (Add Input), turn on a GPU, and open `notebooks/cnn_banglataka.ipynb`.
2. Run all cells. The notebook finds the dataset folder under `/kaggle/input`; if it finds more than one candidate, set
   `DATA_DIR` in the settings cell to the folder of original images.
3. Outside Kaggle: `pip install -r requirements.txt`, download the dataset, and set `DATA_DIR`.

## Repository layout

```
cvpr-banglataka-cnn/
  README.md
  requirements.txt
  .gitignore
  cnn_banglataka.ipynb
```

## Acknowledgements

Dataset: Nuhash and Akter, BanglaTaka (Mendeley Data and Data in Brief, see above).