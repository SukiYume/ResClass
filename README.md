<div align="center">

# ResClass - 使用ResNet的星系与光谱分类

_✨ 天文AI大模型一期 - 预备 &nbsp; | &nbsp; 代码与模型✨_

<img src="https://counter.seku.su/cmoe?name=APOD&theme=r34" /><br>

</div>


## <div align="center">项目训练过程及结果描述</div>

### 关于文件

0. 星系数据 - [Galaxy10.h5](http://www.astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5)、光谱数据 - [train_data_10.fits](https://nadc.china-vo.org/res/file_upload/download?id=46376)
1. 将`Galaxy10.h5`、`train_data_10.fits`和`test_data.fits`文件放入`Data/`文件夹中以正常运行项目
2. 模型测试代码和结果均在`Galaxy_Spec_Predict.ipynb`中展示。

### 关于依赖

0. 除了文件读写外，模型训练只用到了`pytorch`系的依赖。所有依赖写在了`requirements.txt`中，或者也可以直接运行文件，缺啥补啥。

### 星系分类

1. 项目地址 - [SukiYume/ResClass](https://github.com/SukiYume/ResClass)
2. 模型结构 - ResNet50
3. 训练过程 - 因为数据量不大，所以将数据全部载入内存，除了前10个图像外（用作测试），剩余图像按照8:2的比例划分为训练集和验证集。每次从中抽取64张图片进行训练，抽取时根据训练集标签比重加权，图片在训练时会被随机旋转和翻转，以及随机的明亮度对比度变化，训练进行50个epoch。使用下面的命令进行训练。

   ```python
   python galaxy_train.py
   ```
   验证集loss最小时，训练集和验证集的准确率分别是95.37%和86.62%，此时验证集的分类矩阵如下图所示。
   <div align="center"><img src="./logs/galaxy/matrix.png" width="400px"/><br></div>
   验证集准确率最高的一个epoch里，训练集和验证集的准确率分别是95.49%和86.98%。
4. 测试结果 - 选择前10张图像用作测试，这10张图像在训练和验证过程中都没用到，结果在`Galaxy_Spec_Predict.ipynb`中展示。

### 光谱分类

1. 项目地址 - [SukiYume/ResClass](https://github.com/SukiYume/ResClass)
2. 模型结构 - 使用Conv1D构建的ResNet18
3. 训练过程 - 同样数据量不大，全部载入内存（10号fits文件），按照8:2的比例划分为训练集和验证集。每次从中抽取64条光谱进行训练，抽取时根据训练集标签比重加权，光谱在训练时会加入随机噪声以及进行归一化，训练进行50个epoch。使用下面的命令进行训练。

    ```python
   python spec_train.py
   ```
   验证集loss最小时，训练集和验证集的准确率分别是99.96%和99.51%，此时验证集的分类矩阵如下图所示。
   <div align="center"><img src="./logs/spec/matrix.png" width="400px"/><br></div>
   验证集准确率最高的一个epoch里，训练集和验证集的准确率分别是99.99%和99.52%。基本上一个epoch就够用，如果batch_size选更小一点的话。这也是没构建更大一点的ResNet50的原因，因为ResNet18够用了。
4. 测试结果 - 对`test_data.fits`中的数据进行预测，结果写在`Spec_Test.csv`里。

---

<details open>

<summary style="font-size: 1.5em; font-weight: bold;">对天文大模型的理解</summary>
</br>

1. 什么是大模型
   > 堆叠超深的层数，用海量的参数来储存甚至学习到训练数据中的知识
2. 大模型的训练过程
   > 对于大**语言**模型，先使用大量的语料对大模型进行预训练，以期模型能够学到语料中的语法结构以及语料蕴含的知识，此时的模型还不能正常回答提问。需要使用良好组织的问答数据对模型进行监督学习，相当于让模型学会如何输出内容。最后使用训练好的奖励模型对模型进行强化学习，对模型的输出进行打分，相当于让模型学会如何输出好的内容。
3. 计划研究/微调/训练的天文大模型是什么？包括使用什么数据，模型结果，解决的科学问题等
   > 写文章或者打文章草稿的大模型。广泛搜集关于射电暂现源的文章以及综述，至少能够写出一个像样的框架和绪论。
</details>