## Abstract

|             Title | Meta Label Correction for Noisy Label Learning |
| ----------------: | :----------------------------------------------------------- |
|       **Problem** | Leveraging weak or noisy supervision for building effective machine learning models has long been an important research problem.|
|    **Motivation** | 1. Recent work has shown impressive gains by using a meta-learned instance re-weighting approach where a meta-learning framework is used to assign instance weights to noisy labels; <br> 2. One of the limitations of label re-weighting is that it islimited to up or down weighting the contribution of an instance in the learning process. An alternative approach relieson the idea of label correction|
|       **Results** | 1. Run extensive experiments with different label noise levels and types on both image recognition and text classification tasks. <br> 2. Compare the re-weighing and correction approaches showing that the correction framing addresses some of the limitations of re-weighting. We also show that the proposed MLC approach outperforms previous methods in both image and language tasks. 3. show that the proposed MLC approach outperforms previous methods in both image and language tasks.|
| **Contributions** | 1. In this paper, we adopt label correction to address the problem of learning with noisy labels, from a meta-learning perspective. We term our method meta label correction (MLC). <br> 2. Specifically, we view the label correction procedure as a meta-process, which objective is to provide corrected labels for the examples with noisy labels.|
|           **URL** | [AAAI 2021](https://www.aaai.org/AAAI21Papers/AAAI-10188.ZhengG.pdf) |
|     **My Rating** | ★★★★☆                                                    |
|      **Comments** ||

## Overview

### Pipeline

![image](/imgs/MLC_01.png)

![image](/imgs/MLC_02.png)

### Kernel Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans


# ============== mlc step procedure debug with features (gradient-stopped) from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
#
def step_hmlc_K(main_net, main_opt, hard_loss_f,
                meta_net, meta_opt, soft_loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c, 
                eta, args):

    # compute gw for updating meta_net
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)
    gw = torch.autograd.grad(loss_g, main_net.parameters())  # 论文公式(6)的gw
    
    # given current meta net, get corrected label
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)  # 论文公式(6)的H_{alpha, w}，即L_D'{alpha, w}

    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)

    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)   # 论文公式(6)的H_{alpha, w}，即L_D'{apha, w}关于w的一阶偏导

    # f_params_new 为图上w'(alpha)，dparam_s为论文公式(6)的对角矩阵∧
    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True) 
    # 2. set w as w'
    f_param = []
    for i, param in enumerate(main_net.parameters()):
        f_param.append(param.data.clone())
        param.data = f_params_new[i].data # use data only as f_params_new has graph
    
    # training loss Hessian approximation
    Hw = 1 # assume to be identity  ## 论文中的Hw,w，恒等为1

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g)
    loss_g  = hard_loss_f(logit_g, target_g)
    gw_prime = torch.autograd.grad(loss_g, main_net.parameters())  # 图示中的L_D(w')求的梯度gw'

    # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
    tmp1 = [(1-Hw*dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm())**2
    tmp2 = [gw[i]/gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))  # 论文公式(6)的‘-’号前面的项的系数，不包括L_D(w)关于alpha的偏导数

    # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
    Lgw_prime = [ dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]     

    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))  # 论文公式(6)的‘-’号后面的项

    # back prop on alphas
    meta_opt.zero_grad()
    # 前面的f_param_grads为论文文公式(6)的H_{alpha, w}，即L_D'{alpha, w}关于w的一阶偏导，现在关于alpha求偏导，得到L_D'{alpha, w}的二阶导
    # 计算操作后，meta_net有了公式(6)的后一项的梯度，
    proxy_g.backward()  
    
    # accumulate discounted iterative gradient
    for i, param in enumerate(meta_net.parameters()):
        if param.grad is not None:
            param.grad.add_(gamma * args.dw_prev[i])  # 梯度加上求论文公式(6)的‘-’号前面的项，最终完整构成了论文公式(6)
            args.dw_prev[i] = param.grad.clone()

    if (args.steps+1) % (args.gradient_steps)==0: # T steps proceeded by main_net
        meta_opt.step()  # 累计更新meta模型的参数
        args.dw_prev = [0 for param in meta_net.parameters()] # 0 to reset   

    # modify to w, and then do actual update main_net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    main_opt.step()  # 更新main_net
    
    return loss_g, loss_s
```

## BibTex

```
@article{zheng2021meta,
  title={Meta label correction for noisy label learning},
  author={Zheng, Guoqing and Awadallah, Ahmed Hassan and Dumais, Susan},
  journal={AAAI 2021},
  year={2021}
}
```