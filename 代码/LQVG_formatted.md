# ResNetMultiScale

```python
super().__init__()
```

初始化父类 nn.Module，让 PyTorch 能管理这个模块里的网络层。

```python
net = resnet50(weights=ResNet50_Weights.DEFAULT)
stem = net.conv1 + net.bn1 + net.relu + net.maxpool
```

对输入图像做初步特征提取，同时降低图像的空间分辨率。

```python
self.layer1 = net.layer1
self.layer2 = net.layer2
self.layer3 = net.layer3
self.layer4 = net.layer4
```

保留 ResNet 后面的几个阶段。

不同的 self.layer 会输出不同尺度的视觉特征。

其中：

- layer2 输出 c3，对应 stride 8
- layer3 输出 c4，对应 stride 16
- layer4 输出 c5，对应 stride 32

这里的 stride 是 ResNet50 原始结构决定的，不是这段代码重新设置的。

## forward

```python
x = self.stem(x)
```

输入图像先经过 stem，获得初步视觉特征。

```python
x = self.layer1(x)
```

再经过 layer1，继续提取特征。

```python
c3 = self.layer2(x)
c4 = self.layer3(c3)
c5 = self.layer4(c4)
```

实际过程是：

- 输入图像 x
- 经过 stem 得到初步特征
- 经过 layer1 继续提取特征
- 经过已经定义好的 self.layer2 得到 c3
- 经过已经定义好的 self.layer3 得到 c4
- 经过已经定义好的 self.layer4 得到 c5

最后返回：

```python
return [c3, c4, c5]
```

也就是从不同层提取到不同尺度的视觉特征，为后面的跨模态对齐模块和 Transformer 使用。

---

# MLP

一个由多个全连接层组成的小网络，用来把输入特征变成最终输出。

在这份代码里，MLP 主要用作 box_head，也就是把目标特征变成 4 个框坐标。

## init

```python
super().__init__()
```

初始化父类 nn.Module。

```python
input_dim   # 输入维度
hidden_dim  # 隐藏层维度
output_dim  # 输出维度
num_layers  # 一共有多少个 Linear 层
```

```python
dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
```

有 n 个 Linear 层，就需要 n+1 个维度点。

例子：

- input_dim = 256
- hidden_dim = 256
- output_dim = 4
- num_layers = 3

那么：

- dims = [256, 256, 256, 4]

这个列表的意思是：

- 第 1 层：256 -> 256
- 第 2 层：256 -> 256
- 第 3 层：256 -> 4

前面几层保持隐藏特征维度，最后一层变成真正需要的输出维度。

```python
self.layers = nn.ModuleList(
    nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
)
```

这里根据 dims[i] 和 dims[i+1] 创建 Linear 层。

如果 num_layers = 3，就相当于：

```python
self.layers = [
    nn.Linear(256, 256),
    nn.Linear(256, 256),
    nn.Linear(256, 4)
]
```

注意：这里是在 init 里定义好这些层，还没有真正处理数据。

## forward

```python
for i, layer in enumerate(self.layers):
```

依次取出 self.layers 里面的每一个 Linear 层。

```python
x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
```

意思是：

- 如果不是最后一层：x 先经过 Linear 再经过 ReLU
- 如果是最后一层：x 只经过 Linear 不加 ReLU

实际过程：

- 输入特征 x
- 第 1 个 Linear
- ReLU 得到新的 x
- 第 2 个 Linear
- ReLU 得到新的 x
- 第 3 个 Linear 得到最终输出

最后：

```python
return x
```

返回最终输出。

如果这个 MLP 是 box_head，那么输出就是：

```
[B, Q, 4]
```

也就是每个 query 预测一个框，4 个值分别表示：

- cx, cy, w, h

---

# SimpleLQVG

## init

```python
super().__init__()
```

初始化父类 nn.Module。

```python
self.backbone = ResNetMultiScale()
```

定义图像编码器，用 ResNet50 提取多尺度视觉特征。

输出大概是：[c3, c4, c5]

```python
self.input_proj = nn.ModuleList([
    nn.Conv2d(512, hidden_dim, 1),
    nn.Conv2d(1024, hidden_dim, 1),
    nn.Conv2d(2048, hidden_dim, 1),
])
```

input_proj 是视觉特征投影层。

因为 c3、c4、c5 的通道数不同：

- c3: 512
- c4: 1024
- c5: 2048

但是后面的 MSCMA 和 Transformer 需要统一维度，所以用 1x1 Conv 把它们都变成 hidden_dim。

如果 hidden_dim = 256：

- c3: 512  -> 256
- c4: 1024 -> 256
- c5: 2048 -> 256

这里的 1 表示卷积核大小是 1x1。

1x1 Conv 主要改变通道数，不改变特征图的高和宽。

```python
self.text_encoder = AutoModel.from_pretrained(text_model_name)
```

加载预训练文本模型，比如 BERT。

它用来提取词级文本特征。

BERT base 输出的 token embedding 默认是 768 维。

```python
self.text_proj = nn.Linear(768, hidden_dim)
```

把文本特征从 768 维投影到 hidden_dim。

因为 hidden_dim = 256：768 -> 256

这样文本特征和视觉特征就统一到同一个维度，方便后面做跨模态对齐。

```python
self.text_pool = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
)
```

text_pool 用来处理 [CLS] 的文本特征，得到句子级特征。

- Linear 256 -> 256
- Tanh 激活函数

Tanh 会把输出压到 -1 到 1 之间，同时增加非线性表达能力。

```python
self.mscma = MSCMA(hidden_dim, nheads)
```

定义 MSCMA 跨模态对齐模块。

作用是让视觉特征和文本特征先进行交互。

```python
enc_layer = nn.TransformerEncoderLayer(hidden_dim, nheads)
dec_layer = nn.TransformerDecoderLayer(hidden_dim, nheads)
```

定义一层 Transformer Encoder 和一层 Transformer Decoder。

```python
self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
```

把 Encoder layer 和 Decoder layer 堆叠多层。

比如 num_encoder_layers = 4，就是 4 层 Encoder。

```python
self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
```

定义可学习 query embedding。

如果：

- num_queries = 10
- hidden_dim = 256

那么：query_embed = [10, 256]

它的作用是区分不同候选框。

后面会用句子特征生成 language queries，再加上这个可学习 query embedding。

```python
self.class_head = nn.Linear(hidden_dim, 1)
```

分类头。

把每个 query 的特征从 hidden_dim 变成 1 个分数。

这个分数表示当前 query 预测的框是不是目标。

```python
self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
```

框回归头。

用 3 层 MLP，把每个 query 的特征变成 4 个框坐标：cx, cy, w, h

---

# MSCMA

## init

```python
self.text_to_vision = nn.MultiheadAttention(hidden_dim, nheads)
self.vision_to_text = nn.MultiheadAttention(hidden_dim, nheads)
```

定义两个多头注意力层。

**text_to_vision**

- 文本特征作为 query
- 视觉特征作为 key/value
- 作用是让文本去关注图像中相关的位置

**vision_to_text**

- 视觉特征作为 query
- 文本特征作为 key/value
- 作用是让图像位置去关注相关的词

## forward

输入：

- visual_tokens：多尺度视觉特征，是一个 list，每一个 v 的形状是 [HW, B, C]
- text_tokens：文本特征，形状是 [L, B, C]

其中：

- HW = 某一个尺度下的图像位置数量
- L = 文本 token 数量
- B = batch size
- C = hidden_dim

```python
refined_text = text_tokens
refined_visual = []
```

refined_text 先等于原始文本特征。

refined_visual 是空列表，用来存放对齐后的多尺度视觉特征。

```python
for v in visual_tokens:
```

依次取出每一个尺度的视觉特征。

比如：

- 第一次 v = c3_tokens
- 第二次 v = c4_tokens
- 第三次 v = c5_tokens

**LVI 部分**

```python
t2v, _ = self.text_to_vision(refined_text, v, v)
```

这里是文本去看视觉。

在 MultiheadAttention 里：

- query = refined_text
- key = v
- value = v

也就是：文本特征作为 query，视觉特征作为 key/value

含义是：每个文本 token 去图像特征里寻找和自己相关的位置。

t2v 是文本从视觉特征中聚合到的信息。

_ 是 attention 权重，这里没有使用。

```python
refined_text = refined_text * t2v
```

把原来的文本特征和从视觉里得到的信息做逐元素相乘，得到新的文本特征。

也就是：文本特征吸收视觉信息

**VLI 部分**

```python
v2t, _ = self.vision_to_text(v, text_tokens, text_tokens)
```

这里是视觉去看文本。

在 MultiheadAttention 里：

- query = v
- key = text_tokens
- value = text_tokens

也就是：视觉特征作为 query，文本特征作为 key/value

含义是：每个图像位置去文本特征里寻找和自己相关的词。

v2t 是视觉从文本中聚合到的信息。

```python
refined_visual.append(v * v2t)
```

把原始视觉特征 v 和从文本中聚合到的信息 v2t 做逐元素相乘。

然后用 append 加到 refined_visual 里面。

因为视觉特征有多个尺度，所以：refined_visual 依旧是一个 list

最后返回：

```python
return refined_visual, refined_text
```

也就是返回：

- 对齐后的多尺度视觉特征
- 对齐后的文本特征