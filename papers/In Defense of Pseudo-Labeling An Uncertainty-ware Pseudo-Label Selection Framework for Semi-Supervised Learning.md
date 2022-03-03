## Abstract
|             Title | In Defense of Pseudo-Labeling An Uncertainty-ware Pseudo-Label Selection Framework for Semi-Supervised Learning |
| ----------------: | :----------------------------------------------------------- |
|       **Problem** | This work is **in defense of pseudo-labeling**: we demonstratethat pseudo-labeling based methods can perform on par with consistency regularization methods.|
|    **Motivation** | 1.The recent research in semi-supervised learning (SSL) is **mostly dominated by consistency regularization based methods** which achieve strong performance. However, they **heavily rely on domain-specific data augmentations**, which are not easy to generate for all data modalities. <br> 2. Conventional pseudo-labeling based methods achieve **poor results because poor network calibration produces incorrectly pseudo-labeled samples**, leading to noisy training and poor generalization.|
|       **Results** | An uncertainty-aware pseudo-label selection (UPS) framework which improves pseudo labeling accuracy by drastically reducing the amount of noise encountered in the training process. Furthermore, UPS generalizes the pseudo-labeling process, allowing for the creation of negative pseudo-labels; these negative pseudo-labels can be used for multi-label classification as well as negative learning to improve the single-label classification|
|    **Conclusion** | 1. UPS, an uncertainty-aware pseudo-label selection framework that maintains the simplicity, generality, and ease of implementation of pseudo-labeling, while performing on par with consistency regularization based SSL methods. <br> 2. Due to poor neural network calibration, conventional pseudo-labeling methods trained on a large number of incorrect pseudo-labels result in noisy training; our pseudo-label selection process utilizes prediction uncertainty to reduce this noise. |
| **Contributions** | 1. **UPS, a novel uncertainty-aware pseudo-label selection framework** which greatly reduces the effect of poor network calibration on the pseudo-labeling process; <br> 2. While prior SSL methods focus on single-label classification, we generalize pseudo-labeling to create negative labels, allowing for negative learning and multi-label classification; <br> 3. Our comprehensive experimentation shows that the proposed method achieves strong performance on commonly used benchmark datasets CIFAR-10 and CIFAR-100. <br> 4. In addition, we highlight our method’s flexibility by outperforming previous state-of-the-art approaches on the video dataset, UCF-101, and the multi-label Pascal VOC dataset. |
|           **URL** | [ICLR 2021](https://arxiv.org/abs/2101.06329) |
|     **My Rating** | ★★★★☆                                                    |
|      **Comments** |1. 代码逻辑清晰，SSL的主流算法之一的伪标签生成算法中值得借鉴的代码范本，其他伪标签生成算法训练逻辑应该差不多，区别可能是在伪标签生成方式不同，详见代码；<br> 2. 文章主要在现有的伪标签生成算法基础上提出了不确定感知(uncertainty-aware)的伪标签选择，引入了negative pseudo-labeling learning，将置信度低的预测类（即不可能是该类别）加入训练，通过计算negative cross-entropy loss实现；|


### 1. Overview

#### Pseudo-labeling for semi-supervised learning

|Term|Notation|Description|
|:--:|:--:|:--|
|Labeled datset|<!-- $D_L = \lbrace (x^{(i)}, y^{(i)}) \rbrace^{N_L}_{i=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\uh97xdDVPW.svg">|<!-- $N_L$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\ob9BniO9OH.svg"> samples <br> <!-- $x^{(i)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\sSFfaZpiBD.svg">: input <br> <!-- $y^{(i)} = [y_1^{(i)}, ..., y_C^{(i)}] \subseteq \lbrace 0, 1 \rbrace^C$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\dSVQUid7qK.svg">: corresponding label with $C$ class categories|
|Unlabeled dataset|<!-- $D_U = \lbrace x^{(i)} \rbrace^{N_U}_{i=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\jUhlqbOYH9.svg">|<!-- $N_U$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\vBgelvNblL.svg"> samples|

For the unlabeled samples, pseudo-labels <!-- $\tilde{y}^{(i)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\8KKXwHdQZ0.svg"> are generated. Pseudo-labeling based SSL approaches involve learning a parameteriezed model <!-- $f_{\theta}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\6mgo6t9bb5.svg"> trained on dataset <!-- $\tilde{D} = \lbrace (x^{(i)}, \tilde{y}^{(i)}) \rbrace^{N_L + N_U}_{i=1}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\nTAzkf3yTl.svg">.

**Generalizing Pseudo-label Generation**

Let <!-- $p^{(i)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\9eDi9cLu8s.svg"> be the probability outputs of a trained network on the sample <!-- $x^{(i)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\uxy5zPRHws.svg">, such that <!-- $p^{(i)}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\TuCsG0vCyx.svg"> represents the probability of class <!-- $c$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\n14Ds8pBxM.svg"> being present in the sample.

the pseudo-label <!-- $\tilde{y}^{(i)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\suYLjT96rC.svg"> can be generated for <!-- $x^{(i)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\4kuUIjDRqg.svg"> as:

<!-- $$
\tilde{y}^{(i)}_c = \mathbb{1}\left[ p^{(i)}_c \geq \gamma \right]
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\goVb1LX1HZ.svg"></div>

where <!-- $\gamma \in (0, 1]$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\qddBGezcPo.svg">. Note that conventional single-label pseudo-labeling can be derived from the above equation when <!-- $\gamma = \mathop{\max}\limits_{c} p^{(i)}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\QNW0rFpK6v.svg">.

#### Pseudo-label Selection

- Although pseudo-labeling is versatile and modality-agnostic, it achieves relatively poor performance when compared to recent SSL methods. 
- This is due to the large number of incorrectly pseudo-labeled samples used during training. 
- Therefore, we aim at reducing the noise present in training to improve the overall performance. 
- This can be accomplished by intelligently selecting a subset of pseudo-labels which are less noisy; 
- since networks output confidence probabilities for class presence (or class absence), we select those pseudo-labels corresponding with the high-confidence predictions.

Let <!-- $\mathbf{g}^{(i)} = \left[ g^{(i)}_1, ...,  g^{(i)}_C \right] \subseteq \lbrace 0, 1\rbrace^C$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\jYfkaqNG0M.svg"> be a binary vector representing the selected pseudo-labels in sample i, where <!-- $g^{(i)}_c = 1$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\cEdxPg7Npx.svg"> when <!-- $\tilde{y}^{(i)}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\PlhQBXFa1l.svg">  is selected and <!-- $g^{(i)}_c = 0$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\IEqFBUKoIa.svg"> when <!-- $\tilde{y}^{(i)}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\s9METzotbo.svg"> is not selected. This vector is obtained as follows:

<!-- $$
g^{(i)}_c = \mathbb{1} \left[ p^{(i)}_c \geq \tau_p \right] + \mathbb{1} \left[p^{(i)}_c \leq \tau_n \right]
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\Lj517bxn0U.svg"></div>

where <!-- $\tau_p$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\bN8sOjDdI5.svg"> and <!-- $\tau_n$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\3dhDLA0aNh.svg"> are teh confidence thresholds for positive and negative labels(here, <!-- $\tau_p \geq \tau_n$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\pCpi1IzyQh.svg">).

- 注意对于样本$i$，positive label只有一个，而negative label可以有多个；
- 其次，选择了positive label，那么就不选择negative label；

**The parameterized model $f_{\theta}$ is trained on the selected subset of pseudo-labels.**

-  For single-label classification, cross-entropy loss is calculated on samples with selected positive pseudo-labels. 
- If no positive label is selected, then negative learning is performed, using negative cross-entropy loss:

<!-- $$
L_{NCE}(\tilde{y}^{(i)}, \hat{y}^{{i}}, g^{(i)}) = - \frac{1}{s^{(i)}}\sum^C_{c=1}g^{(i)}_c\left( 1 - \tilde{y}^{(i)}_c \right) \log \left( 1 - \hat{y}^{(i)}_c \right)
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\dGvHcSNn3K.svg"></div>

where <!-- $s^{(i)} = \sum_c g^{(i)}_c$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\OPchPZiCCH.svg"> is the number of selected pseudo-labels for sample $i$, <!-- $\hat{y}^{(i)} = f_{\theta}(x^{(i)})$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\q6B1zDs2jq.svg"> is the probability output for the model $f_{\theta}$.

#### Uncertainty-aware Pseudo-label Selection

- Although confidence-based selection reduces pseudo-label error rates, the poor calibration of neural networks renders this solution insufficient - in poorly calibrated networks incorrect predictions can have high confidence scores.
- Propose an uncertainty-aware pseudo-label selection process: by utilizing both the confidence and uncertainty of a network prediction, a more accurate subset of pseudo-labels are used in training.

<!-- $$
g^{(i)}_c = \mathbb{1}\left[ u\left( p^{(i)}_c \leq \kappa_p \right) \right] \mathbb{1}\left[ p^{(i)}_c \geq \tau_p \right] + \mathbb{1}\left[ u\left( p^{(i)}_c \leq \kappa_n \right) \right] \mathbb{1}\left[ p^{(i)}_c \leq \tau_n \right]
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\yHgBHtGWUZ.svg"></div>

where $u(p)$ is the uncertainty of a prediction $p$, and $\kappa_p$ and $\kappa_n$ are the uncertainty thresholds. This additional term, involving $u(p)$, ensures the network prediction is sufficiently certain to be selected.

![image](/imgs/defense_pseudo_labeling_algorithm.png)

训练过程：
1. 第一次迭代：
- 使用有标签的数据训练模型
- 通过该模型对无标签数据进行伪标签预测
- 通过文中的伪标签选择方法选择已有伪标签(positive/negative)的无标签数据的子集加入到下一次的迭代；
2. 第二次迭代：
- 使用有标签的数据以及选择的伪标签(positive label)数据以及交叉熵损失函数训练模型
- 使用文中选择的带有negative label的数据以及负交叉熵(negative cross-entropy)损失函数训练模型
- 使用上述两步得到的模型预测无标签数据的伪标签，并进行伪标签的positive/negative 选择
3. 以此类推

### Kernel Code

#### Training Pipeline

`train-cifar.py`: 算法主要训练过程
```python
    for itr in range(start_itr, args.iterations):
        if itr == 0 and args.n_lbl < 4000: #use a smaller batchsize to increase the number of iterations
            args.batch_size = 64
            args.epochs = 1024
        else:
            args.batch_size = args.batchsize
            args.epochs = args.epchs

        if os.path.exists(f'data/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'):
            lbl_unlbl_split = f'data/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'
        else:
            lbl_unlbl_split = None # 第一次迭代
        
        #load the saved pseudo-labels
        if itr > 0:
            pseudo_lbl_dict = f'{args.out}/pseudo_labeling_iteration_{str(itr)}.pkl'
        else:
            pseudo_lbl_dict = None  # 第一次迭代，仅使用有标签数据济宁训练
        
        # 第一次迭代时进行有/无标签数据划分，其他迭代将有标签数据和有伪标签的数据一起形成lbl_dataset
        # nl_dataset为negative learning
        lbl_dataset, nl_dataset, unlbl_dataset, test_dataset = DATASET_GETTERS[args.dataset]('data/datasets', args.n_lbl,
                                                                lbl_unlbl_split, pseudo_lbl_dict, itr, args.split_txt)  

        model = create_model(args)
        model.to(args.device)

        nl_batchsize = int((float(args.batch_size) * len(nl_dataset))/(len(lbl_dataset) + len(nl_dataset)))

        if itr == 0:
            lbl_batchsize = args.batch_size
            args.iteration = len(lbl_dataset) // args.batch_size
        else:
            lbl_batchsize = args.batch_size - nl_batchsize
            args.iteration = (len(lbl_dataset) + len(nl_dataset)) // args.batch_size

        # 带标签的数据
        lbl_loader = DataLoader(
            lbl_dataset,
            sampler=RandomSampler(lbl_dataset),
            batch_size=lbl_batchsize,
            num_workers=args.num_workers,
            drop_last=True)
        # 带negative pseudo-label的数据
        nl_loader = DataLoader(
            nl_dataset,
            sampler=RandomSampler(nl_dataset),
            batch_size=nl_batchsize,
            num_workers=args.num_workers,
            drop_last=True)

        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)
        # 带positive pseudo-label的数据
        unlbl_loader = DataLoader(
            unlbl_dataset,
            sampler=SequentialSampler(unlbl_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
        args.total_steps = args.epochs * args.iteration
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
        start_epoch = 0

        if args.resume and itr == start_itr and os.path.isdir(args.resume):
            resume_itrs = [int(item.replace('.pth.tar','').split("_")[-1]) for item in resume_files if 'checkpoint_iteration_' in item]
            if len(resume_itrs) > 0:
                checkpoint_itr = max(resume_itrs)
                resume_model = os.path.join(args.resume, f'checkpoint_iteration_{checkpoint_itr}.pth.tar')
                if os.path.isfile(resume_model) and checkpoint_itr == itr:
                    checkpoint = torch.load(resume_model)
                    best_acc = checkpoint['best_acc']
                    start_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])

        model.zero_grad()
        best_acc = 0
        for epoch in range(start_epoch, args.epochs):
            if itr == 0:  # 第一次迭代，仅使用划分的有标签数据
                train_loss = train_initial(args, lbl_loader, model, optimizer, scheduler, epoch, itr)
            else:  # 其他迭代时，使用划分的有标签数据，以及从划分的无标签数据中选择的置信度高的伪标签构成lbl_loader，negative learning使用无标签数据中选择的带有Negative label的数据构成的nl_loader
                train_loss = train_regular(args, lbl_loader, nl_loader, model, optimizer, scheduler, epoch, itr)

            test_loss = 0.0
            test_acc = 0.0
            test_model = model
            if epoch > (args.epochs+1)/2 and epoch%args.test_freq==0:
                test_loss, test_acc = test(args, test_loader, test_model)
            elif epoch == (args.epochs-1):
                test_loss, test_acc = test(args, test_loader, test_model)

            # log记录
            writer.add_scalar('train/1.train_loss', train_loss, (itr*args.epochs)+epoch)
            writer.add_scalar('test/1.test_acc', test_acc, (itr*args.epochs)+epoch)
            writer.add_scalar('test/2.test_loss', test_loss, (itr*args.epochs)+epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            model_to_save = model.module if hasattr(model, "module") else model
            # 保存模型
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out, f'iteration_{str(itr)}')
    
        checkpoint = torch.load(f'{args.out}/checkpoint_iteration_{str(itr)}.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.zero_grad()

        #pseudo-label generation and selection
        # 伪标签生成以及选择
        pl_loss, pl_acc, pl_acc_pos, total_sel_pos, pl_acc_neg, total_sel_neg, unique_sel_neg, pseudo_label_dict = pseudo_labeling(args, unlbl_loader, model, itr)
```

#### DataLoader

`cifar.py`：数据加载类
```python
def get_cifar100(root='data/datasets', n_lbl=10000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    ## augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    if ssl_idx is None:  # 第一次迭代时，划分有/无真实标签的数据
        base_dataset = datasets.CIFAR100(root, train=True, download=True)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 100)
        
        f = open(os.path.join('data/splits', f'cifar100_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:  # 已经有了划分
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:  # 已经有了伪标签
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']  # 文中的g_c二元指示vector
        lbl_idx = np.array(lbl_idx + pseudo_idx)  # 将有真实标签和伪标签的数据合并index

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:  # 第一次迭代没有伪标签
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = CIFAR100SSL(
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None: # 除了第一次迭代，其他迭代期间都有negative learning
        train_nl_dataset = CIFAR100SSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = CIFAR100SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR100(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None: # 其他次数的迭代，包含negative learning以及伪标签postive label参与训练
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else: # 第一次迭代
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset


# 有无真实标签的数据划分
def lbl_unlbl_split(lbls, n_lbl, n_class):
    lbl_per_class = n_lbl // n_class
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    for i in range(n_class):
        idx = np.where(lbls == i)[0]
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        unlbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx, unlbl_idx

# 数据集
class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)  # 真实标签 ground-truth
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))  # nl_mask对应论文的那个g_c二元指示向量，初始化为形状为(n, num_classes)的矩阵
        
        if nl_mask is not None:   # 设置nl_mask
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target  # 设置伪标签

        if indexs is not None: # 根据划分的索引选择数据
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, self.indexs[index], self.nl_mask[index]
```

#### Training Function

`train_util.py`: 训练函数
- `train_initial`: 第一次迭代用到的训练函数，标准的分类训练函数
- `train_regular`: 其他迭代用到的包含negative learning的训练函数，其内容如下：

```python
def train_regular(args, lbl_loader, nl_loader, model, optimizer, scheduler, epoch, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, nl_loader)
    model.train()
    for batch_idx, (data_x, data_nl) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs_x, targets_x, _, nl_mask_x = data_x  # 有标签的数据，注意这里的有标签数据包含原始划分的带真实标签的数据以及有伪标签的数据
        inputs_nl, targets_nl, _, nl_mask_nl = data_nl  # 原始划分的无标签数据集中，带有negative label的数据进行negative learning，注意positive label与negative label的数据不相交

        inputs = torch.cat((inputs_x, inputs_nl)).to(args.device)
        targets = torch.cat((targets_x, targets_nl)).to(args.device)
        nl_mask = torch.cat((nl_mask_x, nl_mask_nl)).to(args.device)

        #network outputs
        logits = model(inputs)

        positive_idx = nl_mask.sum(dim=1) == args.num_classes #the mask for positive learning is all ones
        nl_idx = (nl_mask.sum(dim=1) != args.num_classes) * (nl_mask.sum(dim=1) > 0)  # 带有negative label的索引index
        loss_ce = 0
        loss_nl = 0

        #positive learning
        if sum(positive_idx*1) > 0:
            loss_ce += F.cross_entropy(logits[positive_idx], targets[positive_idx], reduction='mean')  # 标准分类交叉熵损失计算

        #negative learning
        if sum(nl_idx*1) > 0:  # 文中的所示的negative learning训练
            nl_logits = logits[nl_idx]
            pred_nl = F.softmax(nl_logits, dim=1)
            pred_nl = 1 - pred_nl
            pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
            nl_mask = nl_mask[nl_idx]
            y_nl = torch.ones((nl_logits.shape)).to(device=args.device, dtype=logits.dtype)
            loss_nl += torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7)) # 对应文中的negative cross-entropy loss，nl_mask对应文中的g_c

        loss = loss_ce + loss_nl  # positive learning与negative learning的损失相加
        loss.backward()
        losses.update(loss.item())

        # 更新优化器、学习率计划参数、模型参数
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        # tpdm的进度条设置
        if not args.no_progress:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.iterations,
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg
```

#### Pseudo-labeling Algorithm

`pseudo_labeling_util.py`：伪标签生成函数
```python
import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .misc import AverageMeter, accuracy
from .utils import enable_dropout


def pseudo_labeling(args, data_loader, model, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    pseudo_idx = []
    pseudo_target = []
    pseudo_maxstd = []
    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []
    nl_mask = []
    model.eval()
    if not args.no_uncertainty:
        f_pass = 10
        enable_dropout(model)
    else:
        f_pass = 1

    if not args.no_progress:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)  # 划分为无标签数据的真是标签，仅用于训练过程中的伪标签准确度计算，不参与训练和评估
            out_prob = []
            out_prob_nl = []
            for _ in range(f_pass):
                outputs = model(inputs)
                out_prob.append(F.softmax(outputs, dim=1)) #for selecting positive pseudo-labels
                out_prob_nl.append(F.softmax(outputs/args.temp_nl, dim=1)) #for selecting negative pseudo-labels
            out_prob = torch.stack(out_prob)  # (f_pass, batchsize, num_classes)
            out_prob_nl = torch.stack(out_prob_nl)
            out_std = torch.std(out_prob, dim=0)  # (batchsize, num_classes)
            out_std_nl = torch.std(out_prob_nl, dim=0)  # 文中对应的u(p)不确定性度量——方差
            out_prob = torch.mean(out_prob, dim=0)  # (batchsize, )
            out_prob_nl = torch.mean(out_prob_nl, dim=0)  
            max_value, max_idx = torch.max(out_prob, dim=1)  # (batchsize, ), (batchsize, )
            max_std = out_std.gather(1, max_idx.view(-1,1))  # (batchsize, 1)
            out_std_nl = out_std_nl.cpu().numpy()
            
            #selecting negative pseudo-labels
            # 对应文中的negative pseudo-labels的样本的g_c
            interm_nl_mask = ((out_std_nl < args.kappa_n) * (out_prob_nl.cpu().numpy() < args.tau_n)) *1  # (batchsize, num_classes)

            #manually setting the argmax value to zero
            for enum, item in enumerate(max_idx.cpu().numpy()):
                interm_nl_mask[enum, item] = 0
            nl_mask.extend(interm_nl_mask)

            idx_list.extend(indexs.numpy().tolist())  # 全局索引
            gt_list.extend(targets.cpu().numpy().tolist()) # 样本对应的真实标签
            target_list.extend(max_idx.cpu().numpy().tolist()) # 伪标签

            #selecting positive pseudo-labels 伪标签数据选择（positive learning）
            if not args.no_uncertainty:
                selected_idx = (max_value>=args.tau_p) * (max_std.squeeze(1) < args.kappa_p)
            else:
                selected_idx = max_value>=args.tau_p

            # 根据选择索引选择伪标签数据
            pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx].numpy().tolist())
            gt_target.extend(targets[selected_idx].cpu().numpy().tolist())
            
            # 计算损失、准确度
            loss = F.cross_entropy(outputs, targets.to(dtype=torch.long))
            prec1, prec5 = accuracy(outputs[selected_idx], targets[selected_idx], topk=(1, 5))

            # 更新损失和指标
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                data_loader.set_description("Pseudo-Labeling Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(data_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            data_loader.close()

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_idx = np.array(pseudo_idx)

    #class balance the selected pseudo-labels
    if itr < args.class_blnc-1:
        min_count = 5000000 #arbitary large value
        for class_idx in range(args.num_classes):
            class_len = len(np.where(pseudo_target==class_idx)[0])
            if class_len < min_count:
                min_count = class_len
        min_count = max(25, min_count) #this 25 is used to avoid degenarate cases when the minimum count for a certain class is very low

        blnc_idx_list = []
        for class_idx in range(args.num_classes):
            current_class_idx = np.where(pseudo_target==class_idx)
            if len(np.where(pseudo_target==class_idx)[0]) > 0:
                current_class_maxstd = pseudo_maxstd[current_class_idx]
                sorted_maxstd_idx = np.argsort(current_class_maxstd)
                current_class_idx = current_class_idx[0][sorted_maxstd_idx[:min_count]] #select the samples with lowest uncertainty 
                blnc_idx_list.extend(current_class_idx)

        blnc_idx_list = np.array(blnc_idx_list)
        pseudo_target = pseudo_target[blnc_idx_list]
        pseudo_idx = pseudo_idx[blnc_idx_list]
        gt_target = gt_target[blnc_idx_list]

    pseudo_labeling_acc = (pseudo_target == gt_target)*1
    pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100
    print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

    pseudo_nl_mask = []
    pseudo_nl_idx = []
    nl_gt_list = []

    # 带有negative pseudo-label的样本
    # 与positve pseudo-label的样本不重叠
    for i in range(len(idx_list)):
        if idx_list[i] not in pseudo_idx and sum(nl_mask[i]) > 0: # If no positive label is selected, then negative learning is performed, namely select negative label
            pseudo_nl_mask.append(nl_mask[i])
            pseudo_nl_idx.append(idx_list[i])
            nl_gt_list.append(gt_list[i])

    nl_gt_list = np.array(nl_gt_list)
    pseudo_nl_mask = np.array(pseudo_nl_mask)
    one_hot_targets = np.eye(args.num_classes)[nl_gt_list]
    one_hot_targets = one_hot_targets - 1
    one_hot_targets = np.abs(one_hot_targets)
    flat_pseudo_nl_mask = pseudo_nl_mask.reshape(1,-1)[0]
    flat_one_hot_targets = one_hot_targets.reshape(1,-1)[0]
    flat_one_hot_targets = flat_one_hot_targets[np.where(flat_pseudo_nl_mask == 1)]
    flat_pseudo_nl_mask = flat_pseudo_nl_mask[np.where(flat_pseudo_nl_mask == 1)]

    nl_accuracy = (flat_pseudo_nl_mask == flat_one_hot_targets)*1
    nl_accuracy_final = (sum(nl_accuracy)/len(nl_accuracy))*100
    print(f'Pseudo-Labeling Accuracy (negative): {nl_accuracy_final}, Total Selected: {len(nl_accuracy)}, Unique Samples: {len(pseudo_nl_mask)}')
    pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target':pseudo_target.tolist(), 'nl_idx': pseudo_nl_idx, 'nl_mask': pseudo_nl_mask.tolist()}
 
    return losses.avg, top1.avg, pseudo_labeling_acc, len(pseudo_idx), nl_accuracy_final, len(nl_accuracy), len(pseudo_nl_mask), pseudo_label_dict
```