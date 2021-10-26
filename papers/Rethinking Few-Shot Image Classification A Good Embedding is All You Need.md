
## Abstract
|             Title | Rethinking Few-Shot Image Classification A Good Embedding is All You Need? |
| ----------------: | :----------------------------------------------------------- |
|       **Problem** | What matters in few-shot learning?                           |
|    **Motivation** | **[R1]** shows fine-tuning is **only slightly worse** than state-of-the-art algorithms;<br/>**[R2]** shows an improved fine-tuning model performs **slightly worse** than meta-learning algorithms. |
|                   | **[R1]** A baseline for few-shot image classification. ICLR 2020.<br/>**[R2]** A closer look at few-shot classification. ICLR 2019. |
|       **Results** | **Learning** a supervised or self-supervised **representation** on the meta-training set, followed by **training a linear classifier on top of this representation**, **outperforms** state-of-the-art few-shot learning methods, often by **large margins**. |
|    **Conclusion** | Using a **good learned embedding** model can be **more effective** than sophisticated meta-learning algorithms. |
| **Contributions** | 1. A surprisingly simple baseline for few-shot learning, which achieves the **state-of-the-art**;<br/>2. Building upon the simple baseline, use **self-distillation** to further improve performance;<br/>3. Representations learned with state-of-the-art **self-supervised methods** achieve **similar performance** as fully supervised methods |
|           **URL** | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590256.pdf |
|     **My Rating** | ★★★★☆                                                        |
|      **Comments** |                                                              |


## Method

### 1. Overview

####  • Symbols

|Meta-Training Tasks (Sets)|$\mathcal{T} = \lbrace \left(\mathcal{D}^{train}_i, \mathcal{D}^{test}_i \right)\rbrace^I_{i=1}$||
|--:|:--|--|
|**Meta-Testing Tasks (Sets)**|$\mathcal{S} = \lbrace \left(\mathcal{D}^{train}_j, \mathcal{D}^{test}_j \right)\rbrace^J_{j=1}$||
|**Meta-Testing Case**|$\mathcal{D}^{train}=\lbrace\left(x_t,\,y_t\right)\rbrace^T_{t=1}$ <br> $\mathcal{D}^{test}=\lbrace\left(x_q,\,y_q\right)\rbrace^Q_{q=1}$||
|**Embedding Model**|$\Phi_{\ast}=f_{\phi}\left(x_{\ast}\right)$|Backbone, $\ast$ denotes $t$ or $q$|
|**Base Learner $\mathcal{A}$**|$y_{\ast}=f_{\theta}\left(x_{\ast}\right)$|Linear Classifier|

####  • Problem Formulation

##### (1) Meta-Traning :
**The Objective of the Base Learner $\mathcal{A}$ :**
$$\begin{equation}\begin{aligned}
\theta &= \mathcal{A}\left(\mathcal{D}^{train};\phi\right) \\
&= {\underset {\theta} {\operatorname{arg\, min}}}\, \mathcal{L}^{base}\left(\mathcal{D}^{train};\theta,\phi\right) + \mathcal{R}\left(\theta\right)
\end{aligned}\end{equation} \\
$$

where $\mathcal{L}$ is the loss function and $\mathcal{R}$ is the regularization term.

**Average test error of $\mathcal{A}$ on tasks:**

$$
  \phi = {\underset {\theta} {\operatorname{arg\, min}}}\,\mathbb{E}_{\mathcal{T}}\left[\mathcal{L}^{meta}\left(\mathcal{D}^{test};\theta,\phi\right)\right]
$$

where $\theta = \mathcal{A}\left(\mathcal{D}^{train};\phi\right)$
##### (2) Meta-Testing :

**Evaluation of the model :**

$$
\mathbb{E}_{\mathcal{S}}\left[\mathcal{L}^{meta}\left(\mathcal{D}^{test};\theta,\phi\right)\right]
$$

where $\theta = \mathcal{A}\left(\mathcal{D}^{train};\phi\right)$

####  • Method

**Step1**: Merge tasks from meta-training set:

$$\begin{equation}\begin{aligned}
\mathcal{D}^{new} &= \lbrace \left(\mathbf{x}_i,y_i\right)\rbrace^K_{k=1} \\
&= \cup\lbrace\mathcal{D}^{train}_1,...,\mathcal{D}^{train}_i,...,\mathcal{D}^{train}_I\rbrace
\end{aligned}\end{equation}
$$

where $\mathcal{D}^{train}_i$ is the task from $\mathcal{T}$.

**Step2**: **Meta training**, learn a transferrable embedding model $f_{\phi}$, which generalizes to any new task:

$$
\phi = {\underset {\theta} {\operatorname{arg\, min}}} \mathcal{L}^{ce}\left(\mathcal{D}^{new};\phi\right)
$$

$\mathcal{L^{ce}}$ denotes the cross-entropy loss.


**Step3**: **Meta testing**, sample task $\left(\mathcal{D}^{train}_j, \mathcal{D}^{test}_j\right)$ from meta-testing distribution, training base learner (linear classifier), $\theta = \lbrace\mathbf{W},\mathbf{b}\rbrace$:

$$
\theta = {\underset {\lbrace\mathbf{W},\mathbf{b}\rbrace} {\operatorname{arg\, min}}} \sum^{T}_{t=1}\mathcal{L}^{ce}_t\left(\mathbf{W}f_{\phi}\left(\mathbf{x_t}\right)+\mathbf{b}, y_t\right) + \mathcal{R}\left(\mathbf{W},\mathbf{b}\right)
$$

**Step4**:

**(1) Born-again strategy :** distill the knowledge from the embedding model $\phi$ into a new model $\phi^{\prime}$ with an identical architecture:

$$
\phi^{\prime} = {\underset {\phi^{\prime}} {\operatorname{arg\, min}}} \left(\alpha\mathcal{L}^{ce}\left(\mathcal{D}^{new};\phi^{\prime}\right) + \beta KL\left(f\left(\mathcal{D}^{new};\phi^{\prime}\right)\right),f\left(\mathcal{D}^{new};\phi\right)\right)
$$

**(2) Self Distillation :** At each step, the embedding model of k-th generation is trained with knowledge transferred from the embedding model of $(k-1)$-th generation:

$$
\phi^k = {\underset {\phi} {\operatorname{arg\, min}}} \left(\alpha\mathcal{L}^{ce}\left(\mathcal{D}^{new};\phi\right) + \beta KL\left(f\left(\mathcal{D}^{new};\phi\right)\right),f\left(\mathcal{D}^{new};\phi_{k-1}\right)\right)
$$


### 2. Algorithm

**(1) Meta-Training :**

![image](/imgs/RethinkFewShotEmbedding_01.png)

**(2) Meta-Testing :**

![image](/imgs/RethinkFewShotEmbedding_02.png)

### 3. Network Structure

**Backbone :** ResNet-12

**Base Learner :** Multivariate Logistic Regression.

### 4. Code Pipeline

`run.sh`
(1) original code:
```shell
# supervised pre-training
python train_supervised.py --trial pretrain --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# distillation
# setting '-a 1.0' should give simimlar performance
python train_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# evaluation
python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root
```

(2) motified code:
```shell
# supervised pre-training
# python train_supervised.py --trial pretrain --model_path ./checkpoints --tb_path ./tensorboardlogs --data_root ./data/

# distillation
# setting '-a 1.0' should give simimlar performance
# python train_distillation.py -r 0.5 -a 0.5 --path_t ./checkpoints/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --model_path ./dis_checkpoints --tb_path ./dis_tensorboardlogs --data_root ./data/

# evaluation
python eval_fewshot.py --model_path ./dis_checkpoints/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:0.5_b:0_trans_A_born1/resnet12_last.pth --data_root ./data/miniImageNet/

```
### 5. Kernel Code

(1) `train_supervised.py`: The standard pipeline for classification

(2) `train_distillation.py`:
```python
for idx, data in enumerate(train_loader):
    if opt.distill in ['contrast']:
        input, target, index, contrast_idx = data
    else:
        input, target, index = data
    data_time.update(time.time() - end)

    input = input.float()
    if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        if opt.distill in ['contrast']:
            contrast_idx = contrast_idx.cuda()

    # ===================forward=====================
    preact = False
    if opt.distill in ['abound', 'overhaul']:
        preact = True
    feat_s, logit_s = model_s(input, is_feat=True)
    with torch.no_grad():
        feat_t, logit_t = model_t(input, is_feat=True)
        feat_t = [f.detach() for f in feat_t]

    # cls + kl div
    loss_cls = criterion_cls(logit_s, target)
    loss_div = criterion_div(logit_s, logit_t)

    # other kd beyond KL divergence
    if opt.distill == 'kd':
        loss_kd = 0
    elif opt.distill == 'contrast':
        f_s = module_list[1](feat_s[-1])
        f_t = module_list[2](feat_t[-1])
        loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
    elif opt.distill == 'hint':
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        loss_kd = criterion_kd(f_s, f_t)
    elif opt.distill == 'attention':
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_group = criterion_kd(g_s, g_t)
        loss_kd = sum(loss_group)
    else:
        raise NotImplementedError(opt.distill)

    # compute KL loss
    loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

    acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    # ===================backward=====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ===================meters=====================
    batch_time.update(time.time() - end)
    end = time.time()
```

(3) `eval_fewshot.py`:
```python
def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR', opt=None):
    net = net.eval()
    acc = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
            query_xs = query_xs.view(-1, channel, height, width)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            #  clf = SVC(gamma='auto', C=0.1)
            if classifier == 'LR':
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'SVM':
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                                                          C=1,
                                                          kernel='linear',
                                                          decision_function_shape='ovr'))
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)
            elif classifier == 'Proto':
                query_ys_pred = Proto(support_features, support_ys, query_features, opt)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)
```

## Experiments

### 1. Setup

#### **•  Environment**

|     Code | https://github.com/WangYueFt/rfs |
| -------: | -------------------------------------- |
|  **Env** | Pytorch 1.6, CUDA 10.1, Python 3.7,Linux ubuntu 4.15.0-122-generic|
|   **IP** | 122.207.82.54:14000                   |
| **Path** | /homec/xulei/rfs/                   |
|  **GPU** | GeForce RTX 2080Ti, 10G |

#### **•  Datasets**

| Datasets | Description | Url  |
| -------: |:----------- |:---- |
|  Mini-ImageNet         |A standard benchmark for few-shot learning algorithms. It consists of 100 classes randomly sampled from the ImageNet; each class contains 600 downsampled images of size 84 × 84. We follow the widely-used splitting protocol proposed in **R[3]**, which uses 64 classes for meta-training, 16 classes for meta-validation, and 20 classes for meta-testing          |  https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0    |
|   CIFAR-FS       |A derivative of the original CIFAR-100 dataset by randomly splitting 100 classes into 64, 16 and 20 classes for training, validation, and testing, respectively          |  https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0    |
|   Tiered-ImageNet      |Another subset of ImageNet but has more classes (608 classes). These classes are first grouped into 34 higher-level categories, which are further divided into 20 training categories (351 classes), 6 validation categories (97 classes), and 8 testing categories (160 classes). Such construction ensures the training set is distinctive enough from the testing set and makes the problem more challenging.            |  https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0    |
|   FC100       |Also derived from CIFAR-100 dataset in a similar way to tieredImagNnet.          |   https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0   |
|          |   R[3]: Optimization as a model for few-shot learning. ICLR 2017         |      |

#### •  Hyper-Parameters

|         Parameter | Value     |
| ----------------: | --------- |
| **epochs** | 10 |
| **others** | default |

...

### 2. Results


| Dataset |  Metric  | Setup1 |
| :-----: | :------: | :----: |
|   Mini-ImageNet   | test-acc-feat |   0.5597     |  
|         |   test-std   |   0.0079     |

## Comments

- Good Paper

## BibTex

```
@inproceedings{tian2020rethinking,
  title={Rethinking few-shot image classification: a good embedding is all you need?},
  author={Tian, Yonglong and Wang, Yue and Krishnan, Dilip and Tenenbaum, Joshua B and Isola, Phillip},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={266--282},
  year={2020},
  organization={Springer}
}
```
