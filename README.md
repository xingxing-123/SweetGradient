# Sweet Gradient Matters: Designing Consistent and Efficient Estimator for Zero-shot Architecture Search (Neural Networks 2023)

This is an official pytorch implementation for "[Sweet Gradient Matters: Designing Consistent and Efficient Estimator for Zero-shot Architecture Search](https://www.sciencedirect.com/science/article/pii/S0893608023005038)".
![Sweet-img1](img/predictor.png)
![Sweet-img2](img/diagram.png)

# Environment Requirements
* Python 3.6
* PyTorch 1.8.0

# Data Preparation 
The final data format is as follows:

![Sweet-img1](img/data.png)

Here, we provide detailed data acquisition links:
- [cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- [cifar100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [ImageNet16-120](https://drive.google.com/drive/folders/1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7)
- [NAS-Bench-Data/all_graphs.json](https://drive.google.com/drive/folders/18Eia6YuTE5tn5Lis_43h30HYpnF9Ynqf)
- [NAS-Bench-Data/nasbench1_accuracy.p](https://drive.google.com/drive/folders/18Eia6YuTE5tn5Lis_43h30HYpnF9Ynqf)
- [NAS-Bench-Data/NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view)


# Consistency Experiments
To verify the consistency of the experimental results for NAS-Bench-101, please run:
```shell
bash script/Consistency-NB-101.sh
```

To verify the consistency of the experimental results for NAS-Bench-201, please run:
```shell
bash script/Consistency-NB-201.sh
```

# Search Experiments
To verify the search experimental results of NAS-Bench-201, please run:
```shell
bash script/Search-NB-201.sh
```

# Citation
Please cite our paper if you find anything helpful.
```
@article{YANG2023237,
title = {Sweet Gradient matters: Designing consistent and efficient estimator for Zero-shot Architecture Search},
journal = {Neural Networks},
volume = {168},
pages = {237-255},
year = {2023},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2023.09.012},
url = {https://www.sciencedirect.com/science/article/pii/S0893608023005038},
author = {Longxing Yang and Yanxin Fu and Shun Lu and Zihao Sun and Jilin Mei and Wenxiao Zhao and Yu Hu},
}
```

# Acknowledgment
This code is based on [zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas), [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects/tree/main). Great thanks to their contributions.