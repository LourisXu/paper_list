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

### 深度学习（必修）

**pytorch**

**强烈建议除了NLP和RNN的内容，其他全部跟着敲一遍，上机器跑一遍，自己事情自己干，不然项目结构看不懂容易抓瞎，不要想着别人给你写代码，然后你再那里玩╰_╯╬(｀⌒´メ)(｀ι_´メ)(▼へ▼メ)**

[动手学深度学习](https://tangshusen.me/Dive-into-DL-PyTorch/#/)

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


### 详细资料参考

[Deep Learning Tutorial](https://github.com/Mikoto10032/DeepLearning)

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

为了在浏览器中能正确显示markdown文件的公式，我们可以采取如下方式
在chrome的扩展程序中，打开chrome网上应用店，然后搜索MathJax Plugin for Github，下载该插件，并且启用，就可以让上述公式正常显示。

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


