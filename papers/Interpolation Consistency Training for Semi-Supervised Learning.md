## Abstract
|             Title | Interpolation Consistency Training for Semi-Supervised Learning |
| ----------------: | :----------------------------------------------------------- |
|       **Problem** | Consistency regularization methods for semi-supervised learning|
|    **Motivation** | 1. **Random perturbations are inefficient in high dimensions**, as only a tiny proportion of input perturbations are capable of pushing the decision boundary into low-density regions. <br> 2. **The additional computation makes the related methods less appealing in situations** where unlabeled data is available in large quantities. <br> 3. Recent research has shown that **training with adversarial perturbations can hurt generalization performance.**|
|       **Results** | 1. Interpolation Consistency Training (ICT) **encourages  the prediction at an interpolation of unlabeled points to be consistent with the interpolation of the predictions at those points**. <br> 2. ICT **moves  the decision boundary to low-density regions of the data distribution**. <br> 3. ICT achieves state-of-the-art performance when applied to standard neural network architectures on the CIFAR-10 and SVHN benchmark datasets.|
|    **Conclusion** | 1. First, ICT **uses almost no additional computation**, as opposed to computing adversarial perturbations or training generative models. Second, it outperforms strong baselines on two benchmark datasets, even without an extensive hyperparameter tuning. <br> 2. Our theoretical results predicts a failure mode of ICT with low confidence values, which was confirmed in the experiments, providing a practical guidance to use ICT with high confidence values.|
| **Contributions** | 1. Proposed a simple but efficient semi-supervised learning algorithm, Interpolation Consistency Training (ICT) <br> 2. ICT is simpler and more computation efficient than several of the recent SSL algorithms, making it an appealing approach to SSL. <br> 3. Provide a novel theory of ICT to understand how and when ICT can succeed or fail to effectively utilize unlabeled points|
|           **URL** | [IJCAI 2019](https://arxiv.org/abs/1903.03825) |
|     **My Rating** | ★★★★☆      |
|      **Comments** ||

## Overview

### 1. Consistency regularization

- Consistency regularization methods for semi-supervised learning enforce the low-density separation assumption by encouraging invariant prediction <!-- $f(u) = f(u + \delta)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\FAeU1wV6eC.svg"> for perturbations <!-- $u + \delta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\ESMSwVLKC4.svg"> of unlabeled points <!-- $u$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\a3ogTCTIKf.svg">. 
- Such consistency and small prediction error can be satisfied simultaneously if and only if the decision boundary traverses a low-density path.
- **Cluster Assumption**: the existence of cluster structures in the input distribution could hint the separation of samples into different labels. If two samples belong to the same cluster in the input distribution, then they are likely to belong to the same class.
- **Low-density Separation Assumption**: the decision boundary should lie in the low-density regions.

### 2. Interpolation Consistency Training

Given a mixup operation:

<!-- $$
Mix_{\lambda}(a, b) = \lambda \cdot a + (1 - \lambda) \cdot b.
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\gmfO5blICy.svg"></div>

Interpolation Consistency Training (ICT) trains a prediction model <!-- $f_{\theta}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\gNWQqm9FfV.svg"> to provide consistent predictions at interpolations of unlabeled points:

<!-- $$
f_{\theta}(Mix_{\lambda}(u_j, u_k)) \approx Mix_{\lambda}(f_{\theta '}(u_j), f_{\theta '}(u_k))
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\jVI4ZWfy4r.svg"></div>

where <!-- $\theta '$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\BVlpNNO2tg.svg"> is a moving average of <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\oLaMmCZE1y.svg">. 

---

![image](/imgs/Interpolation_consistency_01.png)

- labeled samples <!-- $(x_i, y_i) \sim \mathcal{D}_L$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\I4ltGyLfpE.svg">, drawn from the join distribution <!-- $P_{XY}(X, Y)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\t1GwFlKuoe.svg">.
- unlabeled samples <!-- $u_j, u_k \sim \mathcal{D}_{UL}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\hSWDImkj4U.svg">, drawn from the marginal distribution <!-- $P_X(X) = \frac{P_{XY}(X, Y)}{P_{Y|X}(Y|X)}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\WBshqXBGIB.svg">.

**Our learning goal** is to train a model <!-- $f_{\theta}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\0NT7HrMsMu.svg">, able to predict <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\vkRO7lg1Xe.svg"> from <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\2GS3n6RKR0.svg">. By using stochastic gradient descent, at each iteration <!-- $t$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\NMsGX4Dxl8.svg">, update the parameters <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\5zjVlNqP3k.svg"> to minimize:

<!-- $$
L = L_S + w(t) \cdot L_{US}
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\Y2BejTRKhn.svg"></div>

where <!-- $L_S$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\StLVLb1NcO.svg"> is the usual cross-entropy supervised learning loss over labeled samples <!-- $\mathcal{D}_L$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\5crqQemOC1.svg"> and <!-- $L_{US}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\4kboWZGspl.svg"> is our new interpolation consistency regularization term.

- These two losses are computed on top of (labeled and unlabeled) minibatches, and the ramp function <!-- $w(t)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\leTCRMRgGq.svg"> increases the importance of the consistency regularization term <!-- $L_{US}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\2JIBKyaYzO.svg"> after each iteration.
- **To compute <!-- $L_{US}$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\YUgG3wcMzD.svg">**, sample two minibatches of unlabeled points $u_j$ and $u_k$, and compute their fake labels <!-- $\hat{y}_j = f_{\theta '}(u_j)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\4m4qsXcdgK.svg"> and <!-- $\hat{y}_k = f_{\theta '}(u_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\jkt2sURP1W.svg">, where <!-- $\theta '$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\gJzh69MCFZ.svg"> is an moving average of <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\COc7XbVAIN.svg">.
- Second, compute the interpolation <!-- $u_m = Mix_{\lambda}(u_j, u_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\Hn6sCKgVUf.svg">, as well as the model prediction at that location, <!-- $\hat{y}_m = f_{\theta}(u_m)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\MEgTdk2CP9.svg">.
- Third, update the parameters <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\jFNgBt2j2Z.svg"> as to bring the prediction <!-- $\hat{y}_m$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\hc7hHi06Zs.svg"> closer to the interpolation of the fake labels <!-- $Mix_{\lambda}(\hat{y}_j, \hat{y}_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\CkVVHJceDX.svg">.
- The discrepancy between the prediction <!-- $\hat{y}_m$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\vkHHGVzeKd.svg"> and <!-- $Mix_{\lambda}(\hat{y}_j, \hat{y}_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\tiFg20gCta.svg"> can be measured using any loss: In this paper, use the **mean squared error**.
- On each update, sample a random <!-- $\lambda$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\3rbVt6RPfm.svg"> from <!-- $Beta(\alpha, \alpha)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\D2hJOtIzmP.svg">.

In sum, the population version of ICT term can be written as:

<!-- $$
\mathcal{L}_{US} = \underset {u_j, u_k \sim P_X}{\mathbb{E}} \quad \underset{\lambda \sim Beta(\alpha, \alpha)}{\mathbb{E}} \mathcal{l}(f_{\theta}(Mix_{\lambda}(u_j, u_k)), Mix_{\lambda}(f_{\theta '}(u_j), f_{\theta '}(u_j)))
$$ --> 

<div align="center"><img style="background: white;" src="..\svg\8WcULQ8WWL.svg"></div>

---

![image](/imgs/Interpolation_consistency_02.png)

### 3. Why do interpolations between unlabeled samples provide a good consistency perturbation for semi-supervised training?

(1) To begin with, observe that **the most useful samples on which the consistency regularization should be applied are the samples near the decision boundary**.

- Adding a small perturbation <!-- $\delta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\bIMy9Mauo1.svg"> to such low-margin unlabeled samples <!-- $u_j$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\JX9bsLtpI7.svg"> is likely to push <!-- $u_j + \delta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\Smhpp5hwKG.svg"> over the other side of the decision bounday.
- This would violate the low-density sepration assumption, making <!-- $u_j + \delta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\zzfeOX2Od5.svg"> a good place to apply consistency regularization. These violations do not occur at high-margin unlabeled points that lie far away from the decision boundary.

(2) Consider interpolations <!-- $u_j + \delta = Mix_{\lambda}(u_j, u_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\aWhNNpNGC6.svg"> towards a second randomly selected unlabeled examples $u_k$. Then, the two unlabeled samples $u_j$ and $u_k$ can either:
①. lie in the same cluster
②. lie in different clusters but belong to the same class
③. lie on different clusters and belong to the different classes

Assuming the cluster assumption:
- The probability of ① decreases as the number of classes increases.
- The probability of ② is low if we assume that the number of clusters for each class is balanced.
- The probability of ③ is the highest.

(3) **Assuming that one of <!-- $(u_j, u_k)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\I1DtJssV2i.svg"> lies near the decision boundary (it is a good candidate for enforcing consistency), it is likely (because of the high probability of ③) that the interpolation towards $u_k$ points towards a region of low density, followed by the cluster of the other class.**

Since this is **a good direction to move the decision**, the interpolation is a good perturbation for consistency-based regularization.

Our exposition has argued so far that **interpolations between random unlabeled samples are likely to fall in low-density regions.** 

Thus, such interpolations are good locations where consistency-based regularization could be applied.

## Kernel Code

### (1) Training Pipeline
```python
def train(trainloader,unlabelledloader, model, ema_model, optimizer, epoch, filep):
    global global_step
    
    class_criterion = nn.CrossEntropyLoss().cuda()
    criterion_u= nn.KLDivLoss(reduction='batchmean').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    i = -1
    for (input, target), (u, _) in zip(cycle(trainloader), unlabelledloader):
        # measure data loading time
        i = i+1
        meters.update('data_time', time.time() - end)
        
        if input.shape[0]!= u.shape[0]:
            bt_size = np.minimum(input.shape[0], u.shape[0])
            input = input[0:bt_size]
            target = target[0:bt_size]
            u = u[0:bt_size]
        
        
        if args.dataset == 'cifar10':
            input = apply_zca(input, zca_mean, zca_components)
            u = apply_zca(u, zca_mean, zca_components) 
        lr = adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
        meters.update('lr', optimizer.param_groups[0]['lr'])
        
        if args.mixup_sup_alpha:
            if use_cuda:
                input , target, u  = input.cuda(), target.cuda(), u.cuda()
            input_var, target_var, u_var = Variable(input), Variable(target), Variable(u) 
            
            if args.mixup_hidden: # 隐藏层输出
                output_mixed_l, target_a_var, target_b_var, lam = model(input_var, target_var, mixup_hidden = True,  mixup_alpha = args.mixup_sup_alpha, layers_mix = args.num_mix_layer)
                lam = lam[0]
            else:  # 全连接层输出
                mixed_input, target_a, target_b, lam = mixup_data_sup(input, target, args.mixup_sup_alpha)
                #if use_cuda:
                #    mixed_input, target_a, target_b  = mixed_input.cuda(), target_a.cuda(), target_b.cuda()
                mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
                output_mixed_l = model(mixed_input_var)
                    
            loss_func = mixup_criterion(target_a_var, target_b_var, lam)
            class_loss = loss_func(class_criterion, output_mixed_l)
            
        else:
            input_var = torch.autograd.Variable(input.cuda())
            with torch.no_grad():
                u_var = torch.autograd.Variable(u.cuda())
            target_var = torch.autograd.Variable(target.cuda(async=True))
            output = model(input_var)
            class_loss = class_criterion(output, target_var)
        
        meters.update('class_loss', class_loss.item())
        
        ### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss
        minibatch_size = len(target_var)
        if args.pseudo_label == 'single':
            ema_logit_unlabeled = model(u_var)
            ema_logit_labeled = model(input_var)
        else:
            ema_logit_unlabeled = ema_model(u_var)
            ema_logit_labeled = ema_model(input_var)
        if args.mixup_sup_alpha:
            class_logit = model(input_var)
        else:
            class_logit = output
        cons_logit = model(u_var)

        ema_logit_unlabeled = Variable(ema_logit_unlabeled.detach().data, requires_grad=False)

        #class_loss = class_criterion(class_logit, target_var) / minibatch_size
        
        ema_class_loss = class_criterion(ema_logit_labeled, target_var)# / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())
        
               
        ### get the unsupervised mixup loss###
        if args.mixup_consistency:
                if args.mixup_hidden:
                    #output_u = model(u_var)
                    output_mixed_u, target_a_var, target_b_var, lam = model(u_var, ema_logit_unlabeled, mixup_hidden = True,  mixup_alpha = args.mixup_sup_alpha, layers_mix = args.num_mix_layer)
                    # ema_logit_unlabeled
                    lam = lam[0]
                    mixedup_target = lam * target_a_var + (1 - lam) * target_b_var
                else:
                    #output_u = model(u_var)
                    mixedup_x, mixedup_target, lam = mixup_data(u_var, ema_logit_unlabeled, args.mixup_usup_alpha)
                    #mixedup_x, mixedup_target, lam = mixup_data(u_var, output_u, args.mixup_usup_alpha)
                    output_mixed_u = model(mixedup_x)
                mixup_consistency_loss = consistency_criterion(output_mixed_u, mixedup_target) / minibatch_size# criterion_u(F.log_softmax(output_mixed_u,1), F.softmax(mixedup_target,1))
                meters.update('mixup_cons_loss', mixup_consistency_loss.item())
                if epoch < args.consistency_rampup_starts:
                    mixup_consistency_weight = 0.0
                else:
                    mixup_consistency_weight = get_current_consistency_weight(args.mixup_consistency, epoch, i, len(unlabelledloader))
                meters.update('mixup_cons_weight', mixup_consistency_weight)
                mixup_consistency_loss = mixup_consistency_weight*mixup_consistency_loss
        else:
            mixup_consistency_loss = 0
            meters.update('mixup_cons_loss', 0)
        
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0
        
        
        
        loss = class_loss + mixup_consistency_loss
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit_labeled.data, target_var.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1[0], minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], minibatch_size)
        meters.update('ema_top5', ema_prec5[0], minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(unlabelledloader), meters=meters))
            #print ('lr:',optimizer.param_groups[0]['lr'])
            filep.write(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(unlabelledloader), meters=meters))
    
    train_class_loss_list.append(meters['class_loss'].avg)
    train_ema_class_loss_list.append(meters['ema_class_loss'].avg)
    train_mixup_consistency_loss_list.append(meters['mixup_cons_loss'].avg)
    train_mixup_consistency_coeff_list.append(meters['mixup_cons_weight'].avg)
    train_error_list.append(meters['error1'].avg)
    train_ema_error_list.append(meters['ema_error1'].avg)
    train_lr_list.append(meters['lr'].avg)
```

### (2) Mixup Pipeline

```python
def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    #x, y = x.numpy(), y.numpy()
    #mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])
    
    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    return mixed_x, mixed_y, lam
```

## BibTex

```
@inproceedings{Verma:2019:ICT:3367471.3367546,
 author = {Verma, Vikas and Lamb, Alex and Kannala, Juho and Bengio, Yoshua and Lopez-Paz, David},
 title = {Interpolation Consistency Training for Semi-supervised Learning},
 booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence},
 series = {IJCAI'19},
 year = {2019},
 isbn = {978-0-9992411-4-1},
 location = {Macao, China},
 pages = {3635--3641},
 numpages = {7},
 url = {http://dl.acm.org/citation.cfm?id=3367471.3367546},
 acmid = {3367546},
 publisher = {AAAI Press},
} 
```