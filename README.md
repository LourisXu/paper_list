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

配置清华源，其他阿里源等自行百度
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
scp -P port_number [-R] file_path/dir_path username@ip:dst_dir_path
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
