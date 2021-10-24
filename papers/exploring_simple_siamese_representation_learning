
## Abstract
|             Title | Exploring Simple Siamese Representation Learning |
| ----------------: | ---------------------------------------- |
|       **Problem** |   孪生网络自监督对比学习                  |
|    **Motivation** |   无监督视觉表示学习中孪生网络能够最大限度地提高了同一图像的两个增强之间的相似性，并在一定的条件下避免塌缩解。 |
|       **Results** |    方法在ImageNet和下游任务上取得了有竞争力的结果，并且无须负样本对、大的batches或动量编码器 |
|    **Conclusion** |   孪生网络是一种自然而有效的不变性建模工具，是表征学习的研究热点。最近的方法的孪生结构可能是其有效性的一个核心原因。      |
| **Contributions** | 1. 提出了一种简单的孪生网络自监督对比学习方法  |
|                   | 2. 证明了塌缩解的存在，但是停止梯度更新的操作时避免产生塌缩解的关键   |
|     **My Rating** | ★★★★★                                    |
|      **Comments** | 对近期研究有深入把握，条理清晰，证明严谨  |



## Method

### 1. Overview

Siamese networks have become a common structure in various recent models for unsupervised visual representation learning. These models maximize the similarity between two augmentations of one image, subject to certain
conditions for avoiding collapsing solutions. In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large
batches, (iii) momentum encoders. Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in preventing collapsing. We provide a hypothesis on the implication of stop-gradient, and further show proof-of-concept
experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representation learning

### 2. Algorithm

**Pseudo Code**

![image](/imgs/simsiam_02.png)

**Negative cosine similarity**

$$
\mathcal{D}(p_1, z_2) = -\frac{p_1}{\lVert p_1 \rVert_2} \cdot \frac{z_2}{\lVert z_2 \rVert_2}
$$

**Loss**

$$
\mathcal{L}=\frac{1}{2}\mathcal{D}(p_1, stopgrad(z_2)) + \frac{1}{2}\mathcal{D}(p_2, stopgrad(z_1))
$$

**Kernel Code**

`builder.py`
```python
def forward(self, x1, x2):
    """
    Input:
        x1: first views of images
        x2: second views of images
    Output:
        p1, p2, z1, z2: predictors and targets of the network
        See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
    """

    # compute features for one view
    z1 = self.encoder(x1) # NxC
    z2 = self.encoder(x2) # NxC

    p1 = self.predictor(z1) # NxC
    p2 = self.predictor(z2) # NxC

    return p1, p2, z1.detach(), z2.detach()
```

`main_simsiam.py`
```python
for i, (images, _) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.gpu is not None:
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

    # compute output and loss
    p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    losses.update(loss.item(), images[0].size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3. Network Structure

![image](/imgs/simsiam_01.png)

## Experiments

### 1. Setup

#### **> Environment**

|     Code | https://github.com/facebookresearch/simsiam |
| -------: | -------------------------------------- |
|  **Env** | Pytorch 1.6.0                           |
|   **IP** | 122.207.82.54:14000                   |
| **Path** | /homec/xulei/simsiam/                   |
|  **GPU** | GeForce RTX 2080Ti, 10G |

#### **> Datasets**

| Datasets | Description |
| -------: | ----------- |
|  CIFAR10 | 图像分类数据集 |

**Datasets Load**

（1）原论文使用ImageNet数据集，几百个G
```python
train_dataset = datasets.ImageFolder(traindir, simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))
```

（2）这里为了测试，只使用CIFAR10数据集。
```python
train_dataset = datasets.CIFAR10(root=os.path.join('.', 'Datasets', 'CIFAR10'),
                                 train=True,
                                 download=True,
                                 transform=simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation))
```
（3）自定义数据集

```python
from torch.utils import data as Data


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class CustomDataset(Data.Dataset):
    def __init__(self, imgs_list, imgs_label_list, transform, **kwargs):
        super(CustomDataset, self).__init__(kwargs)
        # achieve your own code for load data
        # for instance, load imgs by 'imgs_list'
        self.imgs_list = imgs_list
        self.label_list = imgs_label_list
        self.transform = transform

    # return the single img (c, w, h)
    def __getitem__(self, item):
        img = self.imgs_list[item]
        label = self.label_list[item]
        return self.transform(img), label

    def __len__(self):
        return len(self.label_list)


imgs_list = []
label_list = []
train_dataset = CustomDataset(imgs_list, label_list, transform=TwoCropsTransform)

# return the batch of imgs (b, c, w, h)
train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True)

```


#### > Hyper-Parameters

|         Parameter | Value     |
| ----------------: | --------- |
| **batchsize** | 32(默认512) |
|   **epochs** | 100 |
|   **momentum**|   0.9        |
|**others**|默认|

#### > Code Pipeline

1. `main_simsiam.py`自监督预训练

```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your imagenet-folder with train and val folders]
```

2. `main_lincls.py`下游分类任务

```
python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/checkpoint_0099.pth.tar \
  --lars \
  [your imagenet-folder with train and val folders]
```

### 2. Results

| Dataset |  Metric  | Setup1 |
| :-----: | :------: | :----: |
|   CIFAR10   | Acc(Top-1) | 52.130 |
|             | Acc(Top-5) | 93.980 |



## Comments

- 源码结构清晰


## BibTex

```
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15750--15758},
  year={2021}
}
```
