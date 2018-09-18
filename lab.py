
# coding: utf-8

# # 英文数字朗读者语音识别

# ## 一、实验介绍
# 
# ### 1 实验内容
# 
# 本实验从什么是自然语言处理讲起，讲解了多个处理模型。并通过调用两个语音数据集进行网络的训练，从而达到使计算机具有处理自然语言的能力。
# 
# ### 2 课程来源
# 
# 本课程源自 [异步社区](http://www.epubit.com.cn/) 的 [《TensorFlow技术解析与实战》](http://www.epubit.com.cn/book/details/4862) 书籍第 $ 11 $ 章，感谢 [异步社区](http://www.epubit.com.cn/) 授权实验楼发布。如需系统的学习本书，请购买[《TensorFlow技术解析与实战》](https://item.jd.com/12073631.html)。
# 
# 为了保证可以在实验楼环境中完成本次实验，我们在原书内容基础上补充了一系列的实验指导，比如实验截图，代码注释，帮助您更好得实战。
# 
# 原书中的训练模型在实验楼环境中运行时间较长，所以本实验选择了书中参考代码的一个内容（与原书的代码内容有不同的地方），在此说明并向代码作者致谢。感兴趣的用户可以通过链接查看其他代码：[https://github.com/pannous/tensorflow-speech-recognition](https://github.com/pannous/tensorflow-speech-recognition)。
# 
# 如果您对于实验有疑惑或者建议可以随时在讨论区中提问，与同学们一起探讨。
# 
# ### 3 实验知识点
# 
# - 自然语言处理的概念
# - 自然语言处理模型
# - 了解梅尔频率倒谱系数
# - 利用 tflearn 构建神经网络
# - 训练神经网络
# 
# ### 4 实验环境
# 
# - python 3
# 
# ### 5 适合人群
# 
# 本课程难度一般，适合具有 $ Python $ 基础的用户。

# ## 二、实验原理

# ### 1 自然语言处理简介
# 
# 自然语言处理是计算机科学领域与人工智能领域中的另一个重要方向，其中很重要的一点就是语音识别（ $ speech\ recognition $ ）。语音识别要解决的问题是让计算机能够 “ 听懂 ” 人类的语音，将语音中包含的文字信息 “ 提取 ” 出来。
# 
# 与语言相关的技术可以应用在很多地方。例如，日本的富国生命保险公司花费 $ 170 $ 万美元安装人工智能系统，把客户的语言转换为文本，并分析这些词是正面的还是负面的。这些自动化工作将帮助人类更快地处理保险业务。除此之外，现在的人工智能公司也在把智能客服作为重点的研究方向。
# 
# 与图像识别不同，在自然语言处理中输入的往往是一段语音或者一段文字，输入数据的长短是不确定的，并且它与上下文有很密切的关系，所以常用的是循环神经网络（ $ recurrent\ neural\ network $ ， $ RNN $ ）模型。

# ### 2 模型的选择
# 
# 下面我们就来介绍使用不同输入和不同数据时，分别适用哪种模型以及如何应用。
# 
# 在下图中，每一个矩形是一个向量，箭头则表示函数（ 如矩阵相乘 ）。最下面一行为输入向量，最上面一行为输出向量，中间一行是 $ RNN $ 的状态。
# 
# ![网络](attachment:%E7%BD%91%E7%BB%9C.png)
# 
# 在上图中从左到右分别表示以下几种情况。
# 
# （1）一对一：没有使用 $ RNN $ ，如 $ Vanilla $ 模型，从固定大小的输入得到固定大小输出（ 应用在图像分类 ）。
# 
# （2）一对多：以序列输出（ 应用在图片描述，输入一张图片输出一段文字序列，这种往往需要 $ CNN $ 和 $ RNN $ 相结合，也就是图像和语言相结合，详见[《TensorFlow技术解析与实战》](https://item.jd.com/12073631.html)第 $ 12 $ 章 ）。
# 
# （3）多对一：以序列输入（ 应用在情感分析，输入一段文字，然后将它分类成积极或者消极情感，如淘宝下某件商品的评论分类 ），如使用 $ LSTM $ 。
# 
# （4）多对多：异步的序列输入和序列输出（ 应用在机器翻译，如一个 $ RNN 读取一条英文语句，然后将它以法语形式输出 ）。
# 
# （5）多对多：同步的序列输入和序列输出（ 应用在视频分类，对视频中每一帧打标记 ）。
# 
# 我们注意到，在上述讲解中，因为中间 $ RNN $ 的状态的部分是固定的，可以多次使用，所以不需要对序列长度进行预先特定约束。更详细的讨论参见 $ Andrej\ Karpathy $ 的文章《[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)》。
# 
# 自然语言处理通常包括语音合成（ 将文字生成语音 ）、语音识别、声纹识别（ 声纹鉴权 ），以及它们的一些扩展应用，以及文本处理，如分词、情感分析、文本挖掘等。

# ### 3 英文数字朗读者识别
# 
# 本节我们就用语音识别的例子来说明 $ TensorFlow $ 在自然语言处理上的应用。这里我们将使用 $ TensorFlow $ 机器学习库，用简短的几行 $ Python $ 代码创建一个超简单的语音识别器。
# 
# 在这个例子中，我们构建一个 $ LSTM $ 循环神经网络，用 $ TFLearn $ 第三方库来训练一个英文数字口语数据集。
# 
# 我们采用 $ spoken\ numbers\ pcm $ 数据集，这个数据集中包含许多人阅读的 $ 0 $ ～ $ 9 $ 这几个数字的英文的音频。分为男声和女声，一段音频（ $ wav $ 文件）中只有一个数字对应的英文的声音。标识方法是“ $ number\_name\_xxx.wav $ ”。如下：
# 
# ```
# 9_Vicki_400.wav
# 9_Victoria_100.wav
# ```
# 
# 下面我们就来训练一个简单的英文口语数字朗读者的识别模型，在实验楼环境上进行训练，经过反复的几次测试，这段代码在实验楼环境中可以在三分钟内达到  $ 98\% $ 的准确率。

# ## 三、开发准备
# 
# 因为数据集下载时间较长，本实验中可将两个数据集( $ spoken\ numbers\ pcm $ 和 $ spoken\ numbers\ spectros 64x64 $ )通过下面的代码进行下载，在 $ shiyanlou $ 根目录中创建 $ Data $ 文件夹，将这两个数据集文件下载至该文件夹中。
# 
# **☞ 示例代码：**

# ```python
# # 下载数据集
# !wget http://labfile.oss.aliyuncs.com/courses/1026/spoken_numbers_pcm.tar
# !wget http://labfile.oss.aliyuncs.com/courses/1026/spoken_numbers_spectros_64x64.tar
# ```

# **☞ 动手练习：**

# 之后需要安装代码运行所依赖的库文件进行，在整个代码实现中我们需要用到 $ tensorflow $ 框架、第三方库 $ tflearn $ 、辅助工具 $ scikit-image $ 、 $ librosa $ 库，其版本分别为 $ tflearn-0.3.2 $ 、 $ tensorflow-1.4.1 $ 、 $ scikit\_image-0.13.1 $ 、 $ librosa-0.5.1 $ 。若 $ tensorflow $ 版本低于 $ 0.9 $ 或高于 $ 1.4.1 $ ，运行时会报错。需要使用命令将这些库文件安装在环境中。
# 
# **☞ 示例代码：**

# ```python
# !pip install -i https://pypi.douban.com/simple/ -U tensorflow==1.4.1
# !pip install -i https://pypi.douban.com/simple/ -U tflearn==0.3.2
# !pip install -i https://pypi.douban.com/simple/ -U scikit_image==0.13.1
# !pip install -i https://pypi.douban.com/simple/ -U librosa==0.5.1
# ```

# 在本次实验中需要调用数据处理文件 $ speech\_data.py $ ，文件可通过下面的代码下载到=根目录下，同时在根目录下创建本次实验文件 $ speech\_train.py $ 。在此次实验中，$ speech\_data.py $ 需要依赖得库文件为 $ scikit-image $ 、 $ librosa $ ，而本次的实验文件需要依赖 $ tensorflow $ 框架和第三方库 $ tflearn $ 。
# 
# **☞ 示例代码：**

# ```python
# !wget http://labfile.oss.aliyuncs.com/courses/1026/speech_data.py
# ```

# 训练模型需要检验的文件以样例的方式给出，实际上该文件可以是样例文件中随意的一个文件，用户可以将训练集中任意文件移动到测试环境下进行检测，通过下面的代码将样例文件 $ 5\_Vicki\_260.wav $ 下载到 $ shiyanlou $ 根目录进行测试。
# 
# **☞ 示例代码：**

# ```python
# !wget http://labfile.oss.aliyuncs.com/courses/1026/5_Vicki_260.wav
# ```

# **☞ 动手练习：**

# ## 四、实验步骤

# ### 1 定义输入数据并预处理数据
# 
# 打开 $ speech\_train.py $ 文件，将下面几个库引入到代码中：
# 
# **☞ 示例代码：**
# 
# ```python
# import os
# import tflearn
# import speech_data as data
# import tensorflow as tf
# %matplotlib inline
# ```
# 
# 将 $ tensorflow $ 的版本号进行打印，并进行源数据的展示。
# 
# **☞ 示例代码：**
# 
# ```python
# print("You are using tensorflow version "+ tf.__version__) #  tflearn version "+ tflearn.version)
# speakers = data.get_speakers()
# number_classes=len(speakers)
# print("speakers",speakers)
# ```
# 
# 之后，需要将语音处理成能够读取的矩阵形式。这里面用到了梅尔频率倒谱系数（ $ Mel\ frequency\ cepstral\ coefficents $ ， $ MFCC $ ）特征向量， $ MFCC $ 是一种在自动语音和说话人识别中广泛使用的特征。
# 
# **☞ 示例代码：**
# 
# ```python
# batch=data.wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
# X,Y=next(batch)
# ```
# 
# 对语言做分帧、取对数、逆矩阵等操作后，生成的 $ MFCC $ 就代表这个语音的特征。可以通过 $ batch\_size $ 调节训练量，越大模型越精确，耗时越长。

# **☞ 动手练习：**

# ### 2 生成模型
# 
# #### 2.1 定义网络模型
# 
# **☞ 示例代码：**
# 
# ```python
# tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
# ```
# 
# 读者会发现，用 $ tflearn $ 真是很简洁，只用五行代码就定义好了一个神经网络模型：
# 
# **☞ 示例代码：**
# 
# ```python
# net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
# net = tflearn.fully_connected(net, 64)
# net = tflearn.dropout(net, 0.5)
# net = tflearn.fully_connected(net, number_classes, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
# ```
# 
# #### 2.2 训练模型
# 
# 接下来训练模型，并把模型存储下来：
# 
# **☞ 示例代码：**
# 
# ```python
# model = tflearn.DNN(net)
# model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
# ```

# **☞ 动手练习：**

# ### 3 预测模型
# 
# 任意输入一个语音文件，进行预测：
# 
# **☞ 示例代码：**
# 
# ```python
# demo_file = "5_Vicki_260.wav"
# demo=data.load_wav_file(data.path + demo_file)
# result=model.predict([demo])
# result=data.one_hot_to_item(result,speakers)
# print("predicted speaker for %s : result = %s "%(demo_file,result)) # ~ 97% correct
# ```
# 
# 结果输出如下：
# 
# ```
# predicted speaker for 5_Vicki_260.wav : result = Vicki
# ```

# **☞ 动手练习：**

# 结果很准确，确实这个音频的数字的朗读者就是 “ $ Vicki $ ”。
# 
# 注：此处输出的结果为神经网络的估计值，并非准确值。又因为实验楼环境限制，网络的训练量级有限，输出的结果可能并不准确。如果希望输出值有更高的精度，可以调整 $ batch\_size $ 为更大的量级，但训练时间成本会大大增加。

# ## 五、实验总结
# 
# 语音识别在现实生活中应用非常广泛，在很多的地方都应用到这项技术。本文介绍了如何利用 $ RNN $ 进行机器学习的模型训练，通过分步进行模型的讲解，使用户了解语音识别神经网络如何建立，并使用该网咯进行模型训练，展示了该模型在语音训练方面的实际应用能力。

# ## 六、扩展阅读
# 
# 本课程源自 [异步社区](http://www.epubit.com.cn/) 的 [《机器学习实战》](https://item.jd.com/12073631.html) 第 $ 11 $ 章，再次感谢 [异步社区](http://www.epubit.com.cn/) 授权实验楼发布。
# 
# 如果学完本课程，对书籍其他内容感兴趣欢迎点击以下链接购买书籍：
# 
# - 立即购买[《机器学习实战》](https://item.jd.com/12073631.html)

# <div style="color: #999;font-size: 12px;font-style: italic;">*本课程内容，由作者授权实验楼发布，未经允许，禁止转载、下载及非法传播。</div>
