
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


### 1. Overview

####  • Symbols

|Meta-Training Tasks (Sets)|<!-- $\mathcal{T} = \lbrace \left(\mathcal{D}^{train}_i, \mathcal{D}^{test}_i \right)\rbrace^I_{i=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\Jjdl3Orp4t.svg">||
|--:|:--|--|
|**Meta-Testing Tasks (Sets)**|<!-- $\mathcal{S} = \lbrace \left(\mathcal{D}^{train}_j, \mathcal{D}^{test}_j \right)\rbrace^J_{j=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\4zcBrqwmW7.svg">||
|**Meta-Testing Case**|<!-- $\mathcal{D}^{train}=\lbrace\left(x_t,\,y_t\right)\rbrace^T_{t=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\MXlR9iDEux.svg"> <br> <!-- $\mathcal{D}^{test}=\lbrace\left(x_q,\,y_q\right)\rbrace^Q_{q=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\Dn9huMunaN.svg">||
|**Embedding Model**|<!-- $\Phi_{\ast}=f_{\phi}\left(x_{\ast}\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\ZASPnp7K3k.svg">|Backbone, <!-- $\ast$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\wF5MwmbrpV.svg"> denotes <!-- $t$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\fSo5uZjxOq.svg"> or <!-- $q$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\VKcwWUQDVk.svg">|
|**Base Learner $\mathcal{A}$**|<!-- $y_{\ast}=f_{\theta}\left(x_{\ast}\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\OFxDa3g6a2.svg">|Linear Classifier|

####  • Problem Formulation

##### (1) Meta-Traning :
**The Objective of the Base Learner $\mathcal{A}$ :**

<!-- $$
\begin{equation}\begin{aligned}
\theta &= \mathcal{A}\left(\mathcal{D}^{train};\phi\right) \\
&= {\underset {\theta} {\operatorname{arg min}}}\, \mathcal{L}^{base}\left(\mathcal{D}^{train};\theta,\phi\right) + \mathcal{R}\left(\theta\right)
\end{aligned}\end{equation}
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\3RP4wTKyU9.svg"></div>

where <!-- $\mathcal{L}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\4kLJtnhlM8.svg"> is the loss function and <!-- $\mathcal{R}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\KyVsOFUmcW.svg"> is the regularization term.

**Average test error of $\mathcal{A}$ on tasks:**

<!-- $$
  \phi = {\underset {\theta} {\operatorname{arg min}}}\,\mathbb{E}_{\mathcal{T}}\left[\mathcal{L}^{meta}\left(\mathcal{D}^{test};\theta,\phi\right)\right]
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\La9Lf3NBA3.svg"></div>

where <!-- $\theta = \mathcal{A}\left(\mathcal{D}^{train};\phi\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\ySkAn3a4z3.svg">

##### (2) Meta-Testing :

**Evaluation of the model :**

<!-- $$
\mathbb{E}_{\mathcal{S}}\left[\mathcal{L}^{meta}\left(\mathcal{D}^{test};\theta,\phi\right)\right]
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\IweY9bbNAB.svg"></div>

where <!-- $\theta = \mathcal{A}\left(\mathcal{D}^{train};\phi\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\GuEyibiO51.svg">

####  • Method

**Step1**: Merge tasks from meta-training set:

<!-- $$
\begin{equation}\begin{aligned}
\mathcal{D}^{new} &= \lbrace \left(\mathbf{x}_i,y_i\right)\rbrace^K_{k=1} \\
&= \cup\lbrace\mathcal{D}^{train}_1,...,\mathcal{D}^{train}_i,...,\mathcal{D}^{train}_I\rbrace
\end{aligned}\end{equation}
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\VR93EXo6SP.svg"></div>

where <!-- $\mathcal{D}^{train}_i$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\8AMnrOcu2a.svg"> is the task from <!-- $\mathcal{T}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\0NsvjHxGCF.svg">.

**Step2**: **Meta training**, learn a transferrable embedding model <!-- $f_{\phi}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\DyKl1DJhME.svg">, which generalizes to any new task:

<!-- $$
\phi = {\underset {\theta} {\operatorname{arg min}}} \mathcal{L}^{ce}\left(\mathcal{D}^{new};\phi\right)
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\MNMNYIPWlp.svg"></div>

<!-- $\mathcal{L^{ce}}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\YUCwTh65vp.svg"> denotes the cross-entropy loss.


**Step3**: **Meta testing**, sample task <!-- $\left(\mathcal{D}^{train}_j, \mathcal{D}^{test}_j\right)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\yxcPVhs9bP.svg"> from meta-testing distribution, training base learner (linear classifier), <!-- $\theta = \lbrace\mathbf{W},\mathbf{b}\rbrace$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\sM9Kq4hH14.svg">:

<!-- $$
\theta = {\underset {\lbrace\mathbf{W},\mathbf{b}\rbrace} {\operatorname{arg min}}} \sum^{T}_{t=1}\mathcal{L}^{ce}_t\left(\mathbf{W}f_{\phi}\left(\mathbf{x_t}\right)+\mathbf{b}, y_t\right) + \mathcal{R}\left(\mathbf{W},\mathbf{b}\right).
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\kTGJLzNCW5.svg"></div>

**Step4**:

**(1) Born-again strategy :** distill the knowledge from the embedding model <!-- $\phi$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\sXkJ5ENb3I.svg"> into a new model <!-- $\phi^{\prime}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\vZ72PXF0ly.svg"> with an identical architecture:

<!-- $$
\phi^{\prime} = {\underset {\phi^{\prime}} {\operatorname{arg min}}} \left(\alpha\mathcal{L}^{ce}\left(\mathcal{D}^{new};\phi^{\prime}\right) + \beta KL\left(f\left(\mathcal{D}^{new};\phi^{\prime}\right)\right),f\left(\mathcal{D}^{new};\phi\right)\right)
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\nk6ZEufUzQ.svg"></div>

**(2) Self Distillation :** At each step, the embedding model of k-th generation is trained with knowledge transferred from the embedding model of <!-- $(k-1)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\a6BypkiNRR.svg">-th generation:

<!-- $$
\phi^k = {\underset {\phi} {\operatorname{arg min}}} \left(\alpha\mathcal{L}^{ce}\left(\mathcal{D}^{new};\phi\right) + \beta KL\left(f\left(\mathcal{D}^{new};\phi\right)\right),f\left(\mathcal{D}^{new};\phi_{k-1}\right)\right)
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\6fnXFhnDcL.svg"></div>

where <!-- $\alpha = 1 - \beta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\wZdihNes4W.svg">.


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

(3) `eval_fewshot.py`: run the meta-testing tasks
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
（4）Datasets

For instance, the `cifar.py` shows the pipepline for loading data:

A. `CIFAR100`: A standard flow for loading data and applying the transform to imgs
```python
class CIFAR100(Dataset):
    """support FC100 and CIFAR-FS"""
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.file_pattern = '%s.pickle'
        else:
            self.file_pattern = '%s.pickle'
        self.data = {}

        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for idx, label in enumerate(labels):
                if label not in label2label:
                    label2label[label] = cur_class
                    cur_class += 1
            new_labels = []
            for idx, label in enumerate(labels):
                new_labels.append(label2label[label])
            self.labels = new_labels

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)
```

B. `MetaCIFAR100`: For meta-learning, the meta-training tasks or meta-testing tasks is divided according to categories, and the categories of meta-training tasks and meta-testing taks are mutually exclusive. To generate a support/query set in a meta-training/meta-testing task, we should select randomly `n-ways` classes from the total categories in the meta-training/meta-testing tasks as a support/query set, where `n-shots` imgs of each class form a support set and all the remaining or `n-queries` imgs of each class form a query set. As the code shown:
```python

class MetaCIFAR100(CIFAR100):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaCIFAR100, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    # Every time this function is called, a meta-training/meta-testing task is generated randomly
    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        # select randomly n-ways classes
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            # select randomly n-shots imgs for the support set
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            # select randomly n-queries imgs for the query set, that must not be contained in the support set
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled) # xor operation
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs  # n_test_runs denotes the number of meta-testing tasks.
```

## Experiments

### 1. Setup

#### **•  Environment**

|     Code | https://github.com/WangYueFt/rfs |
| -------: | -------------------------------------- |
|  **Env** | Ubuntu 16.04.5 LTS, Python 3.7, PyTorch 1.6.0, and cudatoolkit 10.0.130: `conda activate base`|
|   **IP** | 122.207.82.54:14000                   |
| **Path** | /homec/xulei/rfs/                   |
|  **GPU** | GeForce RTX 2080Ti, 10G |

#### **•  Datasets**

| Datasets | Description | Url  |
| ------- |:----------- |:---- |
|  Mini-ImageNet         |A standard benchmark for few-shot learning algorithms. It consists of 100 classes randomly sampled from the ImageNet; each class contains 600 downsampled images of size 84 × 84. We follow the widely-used splitting protocol proposed in **R[3]**, which uses 64 classes for meta-training, 16 classes for meta-validation, and 20 classes for meta-testing          |  https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0    |
|   CIFAR-FS       |A derivative of the original CIFAR-100 dataset by randomly splitting 100 classes into 64, 16 and 20 classes for training, validation, and testing, respectively          |  https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0    |
|   Tiered-ImageNet      |Another subset of ImageNet but has more classes (608 classes). These classes are first grouped into 34 higher-level categories, which are further divided into 20 training categories (351 classes), 6 validation categories (97 classes), and 8 testing categories (160 classes). Such construction ensures the training set is distinctive enough from the testing set and makes the problem more challenging.            |  https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0    |
|   FC100       |Also derived from CIFAR-100 dataset in a similar way to tieredImagNnet.          |   https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0   |
|          |   **R[3]**: Optimization as a model for few-shot learning. ICLR 2017         |      |
|NCT-CRC-HE-100K-NONORM|This is a slightly different version of the "NCT-CRC-HE-100K" image set: This set contains 100,000 images in 9 tissue classes at 0.5 MPP and was created from the same raw data as "NCT-CRC-HE-100K". However, no color normalization was applied to these images. Consequently, staining intensity and color slightly varies between the images. Please note that although this image set was created from the same data as "NCT-CRC-HE-100K", the image regions are not completely identical because the selection of non-overlapping tiles from raw images was a stochastic process.|https://zenodo.org/record/1214456#.YX5-yBpByUk|

#### •  Hyper-Parameters

|         Parameter | Value     |
| ----------------: | --------- |
| **others** | default |

...

### 2. Results

| Dataset |  Metric  | Setup1 |
| :-----: | :------: | :----: |
|5-ways 1-shot||||
|   CIFAR-FS   | test-acc-feat |   0.7508     |  
|         |   test-std   |   0.0080     |
|   MiniImageNet   | test-acc-feat |   0.6278     |  
|         |   test-std   |   0.0070     |
|   FC100   | test-acc-feat |   0.4339     |  
|         |   test-std   |   0.0077     |
|2-ways 1-shot(train:3 val:3 test:3)|||
|NCT-CRC-HE-100K-NONORM|||
|resize: 32 x 32| test-acc-feat |   0.5759     |  
|         |   test-std   |   0.0118     |
|origin: 224 x 224| test-acc-feat |   0.6438     |  
|         |   test-std   |   0.0115     |

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
