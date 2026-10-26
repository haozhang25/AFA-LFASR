# Cascade Residual Learning based Adaptive Feature Aggregation for Light Field Angular Super-Resolution
This is the official PyTorch implementation of the angular SR method in our paper "Cascade Residual Learning based Adaptive Feature Aggregation for Light Field Super-Resolution" accepted in Pattern Recognition, 2025.

## Code

### Dataset
Please download the dataset in the official repository of [DistgASR](https://github.com/YingqianWang/DistgASR).

### Train:
* Run **`Generate_Data_for_Training.m`** to generate training data.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./log/`.

### Test:
* Run `Generate_Data_for_Test.m` to generate test data.
* Run `test_on_datasets.py` to perform network inference.
* The PSNR and SSIM values of each dataset will be saved to `./log/`.
<br><br>

## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{zhang2025cascade,
  title={Cascade residual learning based adaptive feature aggregation for light field super-resolution},
  author={Zhang, Hao and Zhou, Wenhui and Lin, Lili and Lumsdaine, Andrew},
  journal={Pattern Recognition},
  volume={165},
  pages={111616},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement
Our work and implementations are inspired and based on the following project: <br>
[DistgASR](https://github.com/YingqianWang/DistgASR) <br>
[HLFSR-ASR](https://github.com/duongvinh/HLFSR-ASR) <br>
We sincerely thank the authors for sharing their code and amazing research work!

