# cats-vs-dogs
数据集来自 kaggle 上的一个竞赛：[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)，训练集有25000张，猫狗各占一半。测试集12500张，没有标定是猫还是狗。

下面是文件夹的结构，train2里面有两个文件夹，分别是猫和狗，每个文件夹里是12500张图。

```
├── test [12500 images]
├── test.zip
├── test2
│   └── test -> ../test/
├── train [25000 images]
├── train.zip
└── train2
    ├── cat [12500 images]
    └── dog [12500 images]
```

