[toc]
---

> # 科研相关

### 语言学习(先修)

**Python3**

[廖雪峰Python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)：按照例子敲一遍

**Numpy(先修)**

[Numpy快速入门](https://www.numpy.org.cn/user/quickstart.html)

**Pandas**

[菜鸟教程](https://www.runoob.com/pandas/pandas-tutorial.html)
[官网教程](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)

其他统计分析图表库[Matplotlib](https://matplotlib.org/)、[Seaborn](http://seaborn.pydata.org/)等自行百度。

---

### 机器学习

**Sklearn中文教程**：[地址1](https://sklearn.apachecn.org/#/)和[地址2](https://www.scikitlearn.com.cn/)：比较详细，有各种Example，上手快，直接调库即可，要用的时候看对应章节就行。

[Sklearn官网](https://scikit-learn.org/stable/)：有些演示例子以及右上角API查询。

### 深度学习（必修）

**pytorch框架**

**强烈建议跟着其那面CNN、RNN、分类、分割等敲一遍，上机器跑一遍，其他GAN、NLP等领域之后根据自己研究方向跟进，不然项目结构看不懂容易抓瞎**

[动手学深度学习](http://zh.d2l.ai/)：对应深度学习框架[MXNet](https://zh-v2.d2l.ai/d2l-zh.pdf)和[Pytorch](https://zh-v2.d2l.ai/d2l-zh-pytorch.pdf)的实现，建议直接学Pytorch！

框架选择：
- Pytorch科研界广泛使用的框架，轮子多，论文复现快和改进Idea快(推荐)；
- MXNet在显存计算优化上更好，速度快占显存小，但是没有推广起来，API与Pytorch类似；
- Tensorflow早期版本对新手不友好，Keras对其封装后会好很多，工业界Tensorflow还是主流吧。

**理论书籍**

[《统计学习方法》](https://item.jd.com/12522197.html) —— 讲得通透，走科研或算法岗的必修。

[《深度学习》](https://item.jd.com/12128543.html) —— 别名花书，偏理论概念，先学上面动手学深度学习更有用！

[《机器学习》](https://item.jd.com/12762673.html) —— 别名西瓜书，机器学习基础概念以及理论，走科研或算法岗必修。

**注：先看前面Sklearn、Pytorch有代码的Example，搭环境跑Example上手更快，理论后面看个人选择再深入。—— Talk is cheap. Show me the code.**

---

### API查阅

不管是科研还是工作，在学习了书籍的代码后，常常需要翻阅**官网内容**及其对应**代码库的API**(Application Programming Interface)，用以查询各个函数、类等具体定义和实现，学会并掌握API的使用是准确应用的关键。

以Pytorch为例，[Pytorch官网](https://pytorch.org/)首页有基本[安装入口](https://pytorch.org/get-started/locally/)以及[Tutorial](https://pytorch.org/tutorials/)——简单的入门Example，我们关注于[Docs](https://pytorch.org/docs/stable/index.html)，它包括pytorch、torchaudio、torchvision等子模块的API，以[`torch.nn`的API](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)为例，包含了各个参数的说明以及Examples，学习这些Example，最好的方法就是构造输入，根据API测试其输出是否符合预期。

```python
import torch
from torch import nn

x = torch.randn(10, 3, 256, 256) # 生成shape为(10, 3, 256, 256)的张量，模拟批量为10，通道为3、高宽为256像素的图片
layer = nn.Conv2d(in_channels=3, out_channels=4, stride=2)
y = layer(x)
print(y.shape)
```

同时可以查看其API的源码，右上角的[`Source`](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d)，查看具体实现，对于有些情况需要自己查看实现并做修改。

最简单的例子，就是`DataLoader`类的批量随机采样的实现，这是在加载数据常用的类，基本不需要重写，但是有时候需要理解其工作原理才能使得代码执行符合预期，最常见的是自定义`Dataset`类，重写其`__getitem__()`和`__len__()`方法——这在前面《动手学深度学习》后面输入自定义数据集而不是内置数据集时有介绍，不再介绍——而DataLoader内部会根据参数选择合适的`Sampler`，包括`RandomSampler`和`SequentialSampler`，具体实现应用了python的生成器、迭代器等，读者自行查阅。

**这节只是蜻蜓点水说明，目的在于强调API查阅的重要性，望读者不要对英文API说明畏惧**

**不贴图了，自己看链接。**

---

### Latex/Markdown公式

Makrdown公式：

（1）行内：`$ \theta $`

哈哈哈哈 $\theta$ 哈哈哈哈

（2）段内：`$$ \theta $$`

$$\theta$$

**注意markdown公式符号特殊问题：**

（1）下标`_`有时候解析有问题，需要`\_`才生效

（2）符号`&`需要`&amp;`才生效: &amp;

（3）符号`<`需要`&lt;`才生效：&lt;

参考：

[Markdown公式符号](https://www.zybuluo.com/codeep/note/163962)

[Markdown语法](https://www.appinn.com/markdown/)

[Latex公式符号](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)

---

为了在浏览器中能正确显示markdown文件的公式，我们可以采取如下方式
在chrome的扩展程序中，打开chrome网上应用店，然后搜索MathJax Plugin for Github，下载该插件，并且启用，就可以让上述公式正常显示。

---

### 详细资料参考

[Deep Learning Tutorial](https://github.com/Mikoto10032/DeepLearning)

---

### 信号处理

全部以python为语言的实例，matlab我不会。

#### librosa

[librosa](http://librosa.org/doc/latest/install.html)为以python语言为基础的信号处理库，包括常用的MFCCs、Mel滤波器组等，简要Example详见[链接](https://louris.cn/2020/03/21/librosaaudio-processing-library-learning.html/)

[github](https://github.com/search?q=librosa)

服务器安装可能需要安装（需要root权限）:
```
sudo apt-get install libsndfile
```

#### PyWavelet

[PyWavelet](https://pywavelets.readthedocs.io/en/latest/index.html)为以python语言为基础的小波变换处理库，[github](https://github.com/PyWavelets/pywt)。

**[小波变换](https://zhuanlan.zhihu.com/p/22450818)**

**[小波去噪](https://zhuanlan.zhihu.com/p/157540476?utm_source=wechat_session)**

#### Scipy

[Scipy 信号处理](https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html)，包括常见巴特沃斯高通滤波器。

#### nlpaug

[nlpaug](https://nlpaug.readthedocs.io/en/latest/overview/overview.html)为文本、语音增强库，包括基本的语音增强，[Example](https://github.com/makcedward/nlpaug/blob/master/example/audio_augmenter.ipynb)。

**[VTLP语音增强](https://louris.cn/2020/08/28/audio-augmentationvtlp.html/)**

**理论书籍**

[语音信号处理](https://item.jd.com/11950362.html#crumb-wrap)

[语音信号处理实验教程 (附Matlab程序)](https://item.jd.com/11893369.html#crumb-wrap)

[对应python版本](https://github.com/busyyang/python_sound_open)

---

### 顶会论文索引

[AAAI各年会议](https://aaai.org/Conferences/conferences.php)，找对当年的会议官网的对应Accepted Paper，例如[AAAI2021会议录用](https://aaai.org/Conferences/AAAI-21/wp-content/uploads/2020/12/AAAI-21_Accepted-Paper-List.Main_.Technical.Track_.pdf)，然后谷歌学术搜对应的文章。

[CVPR2021](https://openaccess.thecvf.com/CVPR2021?day=all)，对应其他年份改下地址栏。

[会议汇总](https://proceedings.mlr.press/)

[NLP会议汇总](https://aclanthology.org/)

[ACL2021](https://aclanthology.org/events/acl-2021/#2021-acl-long)

其他自己找。

### 中南大学电子图书馆食用方法

[期刊检索](https://lib.csu.edu.cn/)

[SCI期刊检索](https://mjl.clarivate.com/home?PC=K)

[论文数据库](https://lib.csu.edu.cn/)：以[IEEE](http://libdb.csu.edu.cn/resdetail?rid=IEEE)为例，信息门户统一身份认证后就可以免费下载该数据库的论文。具体自己查看

### 中南大学硕/博士论文latex模板

[latex模板](https://github.com/CSUcse/CSUthesis)

按照其说明安装TexStudio编辑器和环境TexLive。

其他编辑器推荐[VSCode](https://code.visualstudio.com/)

VSCode常用插件:
- Markdown Preview Enhanced： Ctrl + Shift + V
- Math to Image：支持右键Markdown的公式转svg

### VPN设置

由于需要访问外网资源，而国内有墙，所以百度搜[Ghelper](http://googlehelper.net/)安装插件。

### 环境配置

#### Python环境的Anaconda安装

安装Anaconda，学习常见环境切换以及依赖包安装
- [官网](https://www.anaconda.com/)
- [清华源镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

#### Conda使用

安装好Anaconda后，学习基本Conda命令使用
- [官网文档](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

1.创建环境

```
conda create --name project_name python=3.x
```
2. 切换环境

```
conda activate project_name
```

3.查看已有环境

```
conda info --envs
```

```
conda environments:

    base           /home/username/Anaconda3
    project_name   * /home/username/Anaconda3/envs/project_name
```

4. 查看改环境下已安装依赖包

```
conda list
```

5. 安装软件包

以安装pytorch为例
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

安装pandas
```
conda install pandas
或者
pip install pandas #看情况使用pip，有些依赖conda没有
```

#### 镜像源

外网资源需要配置镜像
以下配置清华源，其他参见官网链接：

[清华源](https://mirrors.tuna.tsinghua.edu.cn/)

[腾讯云镜像](https://mirrors.cloud.tencent.com/)

[阿里云镜像](https://developer.aliyun.com/mirror/)

##### pip镜像

（1）Linux下，修改`~/.pip/pip.conf`（没有就创建，文件夹'.'表示隐藏文件夹）
内容如下：
```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```
（2）windows下，直接在`C:\Users\xxx\pip`下新建`pip.ini`：
内容如下：
```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn
```

##### Conda镜像

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
conda config --get channels
conda config --remove-key channels
```

##### Apt镜像

`etc/apt/sources/list.d/`下所有文件的cuda和nvida源注释掉，然后修改soruces.list
```
#deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
#deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
#deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
#deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
#deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
#deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
#deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
#deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse
```

#### Vim配置

默认vim查看文件没有显示行号以及通过鼠标滚动翻页，这里设置下`~/.vimrc`
```
set nu
set mouse=a
```

#### 服务器传文件

```
scp -P port_number [-r] file_path/dir_path username@ip:dst_dir_path
```

#### CUDA与CUDNN安装

不需要源码安装，直接敲命令：
- 注意conda安装是在本环境中，切换环境需要重新安装
- 注意cuda与cudnn版本需要对应，具体查看官网[https://developer.nvidia.com/rdp/cudnn-archive]
- 注意cuda安装版本需要与安装的深度学习框架版本对应，以pytorch为例，[官网安装指引](https://pytorch.org/get-started/locally/)
```
conda install cudatoolkit=10.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64
conda install cudnn=7.6.5 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64
```

#### Git设置

##### Git的代理设置

VPN设置：百度`Ghelper`安装后，会员下载PC端VPN然后，设置本地系统代理，WIN10下网络中的代理设置手动代理，以`127.0.0.1:7890`为例
设置代理
```
git config --global http.proxy=http://127.0.0.1:7890
git config --global https.proxy=https://127.0.0.1:7890
```
取消代理
```
git config --global --unset http.proxy

git config --global --unset https.proxy
```

查看git配置
```
git config --global --list
```

---

### 服务器使用若干问题

1. 内存占用高

一般是数据一次性加载到内存导致的，如果考虑每次批次读取数据就不会出现该问题，即每次训练通过dataset去读取对应数据，因为dataset一般是获取第item个数据，然后通过DataLoader批次加载数据。
数据量大的情况下一次性加载内存占用会很高。

2. GPU利用率低

一般是数据加载过程中，不是直接加载训练要的特征，而是在dataset里进行了预处理数据，预处理数据很慢导致GPU一直在等待CPU预处理完的数据导致的。
可以考虑全部提取特征保存后，之后训练直接加载特征，来提高GPU利用率。

3. 指定GPU

```python
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 4'
```
通过os指定GPU，pytorch的指定时在该基础上根据后面的`0,1,4`确定相对顺序的GPU。


---

> # 写在最后

本文只是一点点使用总结，每个部分都有很多可以展开详细的内容，需要读者自己根据需要整理，网上也有很多内容，各自努力吧，少年！

最后送上个人比较喜欢的几句话：
- 选择比努力更重要，但是不努力连选择的权利都没有；
- 请给我一点运气，剩下的我自己努力；

