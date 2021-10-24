
## Abstract
|             Title | Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation |
| ----------------: | ---------------------------------------- |
|       **Problem** | 弱/半监督语义分割掩码——图像归因图（attribution map)生成）                                        |
|    **Motivation** | 分类器得到CAM(Class Attention Map)可以通过对抗爬升技术拓展CAM区域，从而无监督地生成质量可靠的语义分割掩码                                         |
|       **Results** |      比现有弱/半监督方法更优                                    |
|    **Conclusion** |         论文以及代码展示如何使用对抗操作来扩展目标对象的小区分区域，从而获得更好的目标定位                                 |
| **Contributions** | 1.提出了AdvCAM，这是一种图像的归属图，通过操纵它来增加分类得分，允许它识别物体的更多区域                                       |
|                   | 2.实验证明，该方法在不修改或重新训练网络的情况下提高了几种弱监督语义分割方法的性能。                                       |
|                   | 3.在Pascal VOC 2012基准测试中，无论在弱监督语义切分还是半监督语义切分中，我们的方法都取得了显著的性能。                                       |
|     **My Rating** | ★★★★☆                                    |
|      **Comments** | 需要前置预训练的二维图像分类器得到CAM，再通过论文方法拓展                                         |



## Method

### 1. Overview

Weakly supervised semantic segmentation produces a pixel-level localization from a classifier, but it is likely to restrict its focus to a small discriminative region of the target object. AdvCAM is an attribution map of an image that is manipulated to increase the classification score. This manipulation is realized in an anti-adversarial manner, which perturbs the images along pixel gradients in the opposite direction from those used in an adversarial attack. It forces regions initially considered not to be discriminative to become involved in subsequent classifications, and produces attribution maps that successively identify more regions of the target object. In addition, we introduce a new regularization procedure that inhibits the incorrect attribution of regions unrelated to the target object and limits the attributions of the regions that already have high scores. On PASCAL VOC 2012 test images, we achieve mIoUs of 68.0 and 76.9 for weakly and semi-supervised semantic segmentation respectively, which represent a new state-of-the-art

### 2. Algorithm

![image](/imgs/AdvCAM_01.png)

#### Adversarial Climbing

$$
x^t = x^{t - 1} + \xi\nabla_{t - 1} y_{c}^{t - 1} \\
1 \lt t \lt T \\
$$
$x^t$ is the manipluated image at the t-th step,
$y_{c}^{t - 1}$ is the classification logit of $x^{t - 1}$ for class c.

#### Localization Map

$$
\mathcal{A}=\frac{\sum_{t=0}^{T} CAM(x^t)}{max \sum_{t=0}^T CAM(x^t)}
$$

#### Restricting Mask

$$
\mathcal{M}=\mathbb{1}(CAM(x^{t - 1}>\tau))
$$

#### Loss

$$
x^t = x^{t - 1} + \xi\nabla_{t - 1} \mathcal{L}, \\
\mathcal{L} = y_c^{t - 1} - \sum_{k\in\mathcal{C}\backslash c} y_k^{t - 1} - \lambda \lVert \mathcal{M} \odot \lvert CAM(x^{t - 1}) - CAM(x^{0}) \rvert \rVert_1
$$

$\mathcal{C}$ is the set of all classes, $\lambda$ is a hyper-parameter that controls the influence of masking regularization, and $\odot$ is element-wise multiplication.
#### Kernel Code

`obtain_CAM_masking.py`：
```python
# 对抗爬升
def adv_climb(image, epsilon, data_grad):
    sign_data_grad = data_grad / (torch.max(torch.abs(data_grad))+1e-12)
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, image.min().data.cpu().float(), image.max().data.cpu().float()) # min, max from data normalization
    return perturbed_image

# 论文中的 localization map
def add_discriminative(expanded_mask, regions, score_th):
    region_ = regions / regions.max()
    expanded_mask[region_>score_th]=1
    return expanded_mask

# 训练
def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=True)
    cam_sizes = [[], [], [], []] # scale 0,1,2,3
    with cuda.device(process_id):
        model.cuda()
        gcam = GradCAM(model=model, candidate_layers=[args.target_layer])
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            if os.path.exists(os.path.join(args.cam_out_dir, img_name + '.npy')):
                continue
            size = pack['size']
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs_cam = []
            n_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))

            for s_count, size_idx in enumerate([1, 0, 2, 3]):
                orig_img = pack['img'][size_idx].clone()
                for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                    pack['img'][size_idx] = orig_img
                    img_single = pack['img'][size_idx].detach()[0]  # [:, 1]: flip

                    if size_idx != 1:
                        total_adv_iter = args.adv_iter
                    else:
                        if args.adv_iter > 10:
                            total_adv_iter = args.adv_iter // 2
                            mul_for_scale = 2
                        elif args.adv_iter < 6:
                            total_adv_iter = args.adv_iter
                            mul_for_scale = 1
                        else:
                            total_adv_iter = 5
                            mul_for_scale = float(total_adv_iter) / 5

                    for it in range(total_adv_iter):
                        img_single.requires_grad = True

                        outputs = gcam.forward(img_single.cuda(non_blocking=True))

                        if c_idx == 0 and it == 0:
                            cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])

                        gcam.backward(ids=c)

                        regions = gcam.generate(target_layer=args.target_layer)
                        regions = regions[0] + regions[1].flip(-1)

                        if it == 0:
                            init_cam = regions.detach()

                        cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                        logit = outputs
                        logit = F.relu(logit)
                        logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]

                        valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                        logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)

                        expanded_mask = torch.zeros(regions.shape)
                        expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)
                        # loss计算
                        L_AD = torch.sum((torch.abs(regions - init_cam))*expanded_mask.cuda())
                        # loss
                        loss = - logit_loss - L_AD * args.AD_coeff

                        model.zero_grad()
                        img_single.grad.zero_()
                        loss.backward()

                        data_grad = img_single.grad.data

                        perturbed_data = adv_climb(img_single, args.AD_stepsize, data_grad)
                        img_single = perturbed_data.detach()

                outputs_cam.append(cam_all_classes)

            # CAM计算
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs_cam]), 0)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs_cam]

            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
```

### 3. Network Structure

本文是在[CVPR 2019 IRN 无监督实例分割](https://github.com/jiwoon-ahn/irn)的基础上发展而来，具体无监督训练的网络结构需要参考该篇文章，有兴趣的同学可以深入了解。


## Experiments

### 1. Setup

#### **> Environment**

|     Code | https://github.com/jbeomlee93/AdvCAM |
| -------: | -------------------------------------- |
|  **Env** | Pytorch 1.6.0                           |
|   **IP** | 122.207.82.54:14000                   |
| **Path** | /homec/xulei/advcam/                   |
|  **GPU** | GeForce RTX 2080Ti, 10G |

#### **> Datasets**

| Datasets | Description |
| -------: | ----------- |
|     VoC2012     |  图像分割benchmark数据集|           |


#### > Hyper-Parameters

|         Parameter | Value     |
| ----------------: | --------- |
| **irn_crop_size** |224(默认512)|
|         **other** |   默认    |         

#### > Code PipeLine

`bash get_mask_quality.sh`
```
#!/bin/bash

python obtain_CAM_masking.py --train_list voc12/train_aug.txt
python run_sample.py --eval_cam_pass True --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True

```

`obtain_CAM_masking.py` -> `run_sample.py`

1. obtain_CAM_masking获取论文方法得到的CAM；
2. run_sample进行后续处理，生成掩码，其参数：
- train_irn_pass: 训练irn
- make_seg_seg_pass：生成掩码mask
- eval_sem_seg_pass：评估掩码质量

### 2. Results

| Dataset |  Metric  | Setup1 |
| :-----: | :------: | :----: |
| VoC 2012|   mIoU   | 0.66823|


## Comments

- 源码结构清晰
- 目前只关注其核心生成CAM的方法，前置IRN未深入以及下游任务做分割、检测等需要该领域同学深入。


## BibTex

```
@inproceedings{lee2021anti,
  title={Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation},
  author={Lee, Jungbeom and Kim, Eunji and Yoon, Sungroh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4071--4080},
  year={2021}
}
```
