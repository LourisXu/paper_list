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

### Overview

#### 1. Consistency regularization

- Consistency regularization methods for semi-supervised learning enforce the low-density separation assumption by encouraging invariant prediction <!-- $f(u) = f(u + \delta)$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\FAeU1wV6eC.svg"> for perturbations <!-- $u + \delta$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\ESMSwVLKC4.svg"> of unlabeled points <!-- $u$ --> <img style="transform: translateY(0.1em); background: white;" src="..\svg\a3ogTCTIKf.svg">. 
- Such consistency and small prediction error can be satisfied simultaneously if and only if the decision boundary traverses a low-density path.
- **Cluster Assumption**: the existence of cluster structures in the input distribution could hint the separation of samples into different labels. If two samples belong to the same cluster in the input distribution, then they are likely to belong to the same class.
- **Low-density Separation Assumption**: the decision boundary should lie in the low-density regions.

#### 2. Interpolation Consistency Training

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

#### 3. Why do interpolations between unlabeled samples provide a good consistency perturbation for semi-supervised training?

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