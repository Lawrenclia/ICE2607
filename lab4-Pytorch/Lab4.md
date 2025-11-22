# Lab4 实验报告

- **姓名：蒋昊翔**
- **学号：524030910083**
- **班级：电院2401**

---



## 一、实验概览

### 1. 实验 4.1：CIFAR-10 图片分类

- 实验目的：本次实验的目的是学习和入门 PyTorch 深度学习框架。
- 实验原理：通过 PyTorch 构建一个卷积神经网络（CNN）模型（ResNet20），在 CIFAR-10 图像数据集上对其进行训练。理解深度学习的训练过程，包括损失函数（Loss）、反向传播和优化器（Optimizer）如何工作，最终目标是让模型能够准确区分10个类别的图像。



### 2. 实验 4.2：Search by CNN features (图像检索)

- 实验目的：本次实验的目的是利用预训练的深度学习模型（如 ResNet50）提取图像的高维语义特征。
- 实验原理：与 Lab3 中基于颜色直方图的低级特征不同，CNN 模型在大型数据集（如 ImageNet）上预训练后，其倒数第二层输出的特征向量包含了丰富的语义信息。实验利用这种特性，通过计算特征向量之间的相似度（如向量夹角），来实现一个“以图搜图”的图像检索系统。



### 3. 实验环境

- 主要库：`torch`, `torchvision`, `torch.nn`, `torch.optim`, `os`, `numpy`, `PIL`, `matplotlib`

- 文件结构：

  ```
  ├── lab4_1.py                   # 任务4.1代码内容  
  ├── lab4_2.py                   # 任务4.2代码内容
  ├── models.py                   # 任务4.1依赖的模型
  ├── query/                      # 用于检索的图片
  │   ├── query_airplane.jpg   
  │   ├── …………
  │   └── query_panda.jpg    
  ├── image_library/              # lab4_2图片库
  │   ├── airplane/    
  │   │   ├── Image_1.jpg                
  │   │   ├── …………
  │   ├── …………
  │   └── tree/
  │       ├── Image_1.jpg                
  │       ├── …………       
  └── checkpoint/                 # lab4_1结果
  │   ├── ckpt_0_acc_xxx.pth        
  │   ├── …………
  │   ├── ckpt_9_acc_xxx.pth    
  │   └── final.pth               # lab4_1最终训练结果    
  └── output/                     # lab4_2结果示意图
  ```

------



## 二、 练习题的解决思路

### 1. 实验 4.1 解决思路

这个实验主要就是实现一个标准的 PyTorch 训练和测试流程，关键是 `train(epoch)`、`test(epoch)` 这两个函数。

- `train(epoch)` 函数： 这个函数负责跑一个 epoch 的训练。
  1. 用 `model.train()` 打开训练模式。
  2. 循环读取 `trainloader` 里的每一批数据（batch）。
  3. `optimizer.zero_grad()`：清空上次的梯度。
  4. `outputs = model(inputs)`：模型算一下结果。
  5. `loss = criterion(...)`：计算损失。
  6. `loss.backward()`：反向传播，算出梯度。
  7. `optimizer.step()`：更新模型参数。
  8. 打印出当前的 epoch、loss 和训练准确率。
- `test(epoch)` 函数： 这个函数在每个 epoch 训练完后，用来在独立的测试集上检查模型的效果。
  1. `model.eval()`：切换到评估模式。
  2. `with torch.no_grad():`：告诉 PyTorch 这部分不用算梯度。
  3. 循环读取 `testloader` 里的数据。
  4. 算一下模型在测试集上的总准确率 `acc`。
  5. `torch.save(...)`：把当前的模型状态（`state`）存成 `.pth` 文件，文件名里带上 epoch 和准确率。
- 主训练循环： 最后是一个 `for epoch in range(0, 10):` 循环，跑 10 个 epoch。
  1. 调学习率：里面加了个 `if` 判断，在 `epoch == 5` 的时候，把学习率（lr）从 0.1 降到 0.01。
  2. 存储模型：在 `test` 函数里加了个判断，在最后一个 epoch（`epoch == 9`）时，把模型额外存一个叫 `final.pth` 的文件。



### 2. 实验 4.2 解决思路

为了让 `lab4_2.py` 能有数据去检索，我们首先需要一个“图像库”（Library）。 `download.py` 脚本可以完成这个任务（可见源代码部分）。

- `download_images_task()` 函数：
  1. 创建目录：脚本首先会检查 `image_library` 这个目录存不存在，如果不存在，就创建一个。
  2. 定义类别：定义了一个包含 10 个类别（queries）的列表，比如 "panda", "cat", "dog", "airplane" 等。
  3. 下载图片：使用 `bing_image_downloader` 这个库，循环遍历每个类别，每个类别下载 10 张图片。
  4. 保存位置：`downloader.download` 函数会把下载的图片自动保存在 `image_library` 目录下，并且按类别分好子文件夹。
- 主程序：
  - 直接运行 `download.py` 就会调用 `download_images_task()`，开始下载全部 10 个类别，总计约 100 张图片，构建出 `lab4_2.py` 所需的图像库。



这个实验是搭一个搜图系统，主要由下面几个函数实现：

- `features(x)` 函数： 这个函数是用来从 ResNet50 里拿特征的。它按顺序跑了 ResNet50 的大部分层（从 `conv1` 到 `avgpool`），就是在最后的全连接层（`fc`）之前停住，拿到了 2048 维的特征向量。PPT 里也提到了，我们通常把模型最终分类的前一层当作图像特征。
- `extract_feature(image_path, ...)` 函数： 这是一个帮助函数，处理一张图片，并抽取出特征。
  1. 加载图片 `default_loader`。
  2. 用 `trans` 变换（缩放、裁剪、归一化）来处理图片，让它符合 ResNet50 的输入要求。
  3. 在 `torch.no_grad()` 下，用 `features()` 函数算出特征。
  4. L2 归一化：最后一步很重要， `feature_np / np.linalg.norm(feature_np)`。把特征向量的长度都变成 1。
- `build_database(library_dir, ...)` 函数： 这个函数是建特征库，就是把 `image_library` 文件夹里所有图片的特征都提前算好。
  1. 用 `os.walk` 遍历 `library_dir` 里的所有子文件夹和图片。
  2. 对每张图片，都调用 `extract_feature()` 算出特征。
  3. 把所有特征存进 `db_features` 列表，把对应的路径存进 `db_image_paths` 列表。
  4. 最后返回一个存着所有特征的 NumPy 数组，和一个存着所有路径的列表。
- `plot_results(query_path, ...)` 函数： 这个函数是用来把搜索结果画出来，看着更直观。
  1. 用 `plt.subplots(1, 6, ...)` 建一个 1x6 的图。
  2. `axes[0, 0]` (左上角) 放要搜的图（Query Image）。
  3. 剩下 5 个格子，按顺序放 Top-5 的搜索结果。
  4. 给每张图加上标题，写上排名、得分和文件名。
  5. `plt.savefig()` 把这张对比图保存下来。
- 主程序逻辑 (`if __name__ == "__main__":`)： 程序的入口，负责把所有事串起来。
  1. 先加载 ResNet50 模型。
  2. 调用 `build_database()`，把特征库建好。
  3. 遍历 `query` 文件夹里所有要搜的图。
  4. 对每一张要搜的图：
  5. 调用 `extract_feature()` 算它的特征 `query_feature`。
  6. 算相似度：`np.dot(db_features, query_feature)`。这一步就是用查询特征和库里所有特征算点积，速度很快。因为我们前面做了 L2 归一化，所以算点积就等于在算余弦相似度（向量夹角）。
  7. 排序：`np.argsort(scores)[::-1][:TOP_K]`，拿到分数最高的前 K 个的索引。
  8. 打印出 Top-5 结果的路径和分数。
  9. 调用 `plot_results()` 把结果图画出来并保存。

------



## 三、代码运行结果

### 1. 实验 4.1 运行结果

`lab4_1.py` 按照 10 轮策略训练。训练结束后，在 `./checkpoint/` 文件夹中生成了 10 个模型文件以及最后一次训练结果`final.pth`。

根据上述文件名，我们提取出每个 epoch 结束时在测试集上的准确率（Test Acc），汇总如下表：

| **Epoch** | **学习率 (LR)** | **文件名**                 | **Test Acc (%)** |
| --------- | --------------- | -------------------------- | ---------------- |
| 0         | 0.1             | `ckpt_0_acc_45.700000.pth` | 45.70            |
| 1         | 0.1             | `ckpt_1_acc_52.850000.pth` | 52.85            |
| 2         | 0.1             | `ckpt_2_acc_53.620000.pth` | 53.62            |
| 3         | 0.1             | `ckpt_3_acc_59.500000.pth` | 59.50            |
| 4         | 0.1             | `ckpt_4_acc_64.420000.pth` | 64.42            |
| 5         | 0.01            | `ckpt_5_acc_74.820000.pth` | 74.82            |
| 6         | 0.01            | `ckpt_6_acc_75.510000.pth` | 75.51            |
| 7         | 0.01            | `ckpt_7_acc_76.080000.pth` | 76.08            |
| 8         | 0.01            | `ckpt_8_acc_75.930000.pth` | 75.93            |
| 9         | 0.01            | `ckpt_9_acc_76.390000.pth` | 76.39            |

基于此，绘制出epoch和测试准确率的关系图，如下：

 <div style="display: flex; justify-content: center; gap: 0px;"> 
     <img src="D:\Files\openCV\lab4\codes\epoch_vs_acc_plot.png" width="800" alt="epoch_acc"/> 
</div>


### 2. 实验 4.2 运行结果

使用一个包含 80 张图片的 `image_library` 图片库（分成10类，每类有8张图片）和5张 `query` 图片进行测试。

测试结果如下（控制台输出内容和图片内容一致）：

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab4\codes\output\query_airplane_result.png" width="1000" alt="airplane"/>
</div>

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab4\codes\output\query_building_result.png" width="1000" alt="building"/>
</div>

<div style="display: flex; justify-content: center; gap: 0px;">
	<img src="D:\Files\openCV\lab4\codes\output\query_cat_result.png" width="1000" alt="cat"/>
</div>

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab4\codes\output\query_dog_result.png" width="1000" alt="dog"/>
</div>

<div style="display: flex; justify-content: center; gap: 0px;">
	<img src="D:\Files\openCV\lab4\codes\output\query_panda_result.png" width="1000" alt="panda"/>
</div>

------



## 四、实验结果分析与思考

### 1. 实验 4.1 分析与思考

- Test Acc 趋势： 从 “Epoch 与 Accuracy 关系图” 能看出来，Test Acc 整体是上升的。这说明模型确实在学习，并且在新数据上的表现越来越好。

- 思考题：Train Acc 与 Test Acc 的关联与不同
	- 不同点：
    - Train Acc（训练准确率） 反映的是模型在训练集（Training Set）上的表现，即模型对“学习”过的数据的拟合程度。
    - Test Acc（测试准确率） 反映的是模型在测试集（Test Set）上的表现，即模型对“未见过”的新数据的预测能力。
  - 关联：
    - Test Acc 是衡量模型 “泛化能力” 的真正标准，即模型是否能举一反三。
    - 在训练过程中，Train Acc 通常会高于 Test Acc。如果 Train Acc 远高于 Test Acc（例如，Train Acc 接近 100%，而 Test Acc 停滞在 70%），则表明模型发生了 “过拟合” ，它仅仅是 “背会” 了训练数据，而没有学到通用的规律。

- 思考题：学习率 (LR) 从 0.1 变到 0.01 的影响

  - 现象：
    - 根据实验结果，在 Epoch 5 时，学习率 (LR) 从 0.1 调整至 0.01。
    - Test Acc 立即发生了一次显著的跳升，在我们的实验中是从 64.42% 跃升至 74.82%。

  - 原因分析：
    - LR = 0.1（大学习率）：在训练初期，大学习率有助于模型快速收敛，让损失（Loss）迅速下降，快速找到最优解所在的“大致区域”。
    - LR = 0.01（小学习率）：当训练进行到后期，模型已经接近“谷底”（最优解）。此时如果仍使用 0.1 的大学习率，模型参数可能会在“谷底”附近“来回震荡”，难以收敛到最低点。切换到 0.01 的小学习率，就好比从“大步快跑”切换到“小步微调”，允许模型在已找到的“山谷”附近进行更精细的搜索，从而收敛到更优的解，因此准确率会得到明显提升。



### 2. 实验 4.2 分析与思考

- 检索结果准确性：

  从结果图来看，检索系统的准确率非常高。例如，当查询 `query_airplane.jpg` 时，返回的 Top-5 结果全部是 `image_library/airplane/` 目录下的飞机图片，并且相似度得分（Score）都处于高位。这有力地证明了 ResNet50 提取的 2048 维特征向量确实抓住了图像的“语义”或“内容”，而不是像 Lab3 中那样只抓住了“颜色”。

- 可视化结果的必要性：

  `plot_results` 函数带来的可视化，极大地提升了实验的直观性。相比于只在控制台打印一长串文件路径，这种对比图（最左侧为查询图，其余为 Top-5 结果）能让我们 “一目了然” 地判断检索效果的好坏。

- 高级特征 (Lab4) vs 低级特征 (Lab3)：

  这次实验和 Lab3 形成了鲜明的对比。Lab3 的颜色直方图是一种“低级”特征，它只关心全局的颜色分布，但不在乎这些蓝色是“天空”还是“海洋”。而本实验的 CNN 特征是“高级”的语义特征，它是在 ImageNet 大数据集上训练出来的，模型“理解”图像的内容。因此，它能区分“飞机”和“汽车”，实现真正的“以内容搜图”，效果远非 Lab3 可比。

- 检索效率与扩展性思考：

  - 建库 (`build_database`)：这是一个 $O(N)$ 的过程（$N$ 为图像库中图片数量），并且计算量大，因为每张图片都要跑一次 CNN 前向传播。但好处是，这个过程只需要执行一次。
  - 检索 (`search`)：这也是一个 $O(N)$ 的过程。`np.dot` 操作需要计算查询向量与库中 $N$ 个特征向量的点积。
  - 思考：在本次实验中 $N=80$，这个 $O(N)$ 的检索几乎是瞬时的。但如果 $N$ 达到几百万或几千万，每次检索都要遍历几千万个向量是无法接受的。届时，这个 $O(N)$ 的线性扫描将成为瓶颈，我们就必须引入 Lab3 中提到的 LSH 或其他 ANN（近似近邻）算法，为这 2048 维的特征建立索引，将查询效率从 $O(N)$ 优化到 $O(log N)$ 甚至 $O(1)$。

------



## 五、实验感想

### 1. 实验 4.1 感想

通过补充 `lab4_1.py` 的代码，我跑通了一个完整的深度学习训练流程。我现在更清楚地理解了 `DataLoader` (加载数据)、`ResNet20` (模型)、`CrossEntropyLoss` (算损失) 和 `SGD` (优化器) 是如何协同工作的。

特别是 `epoch`、`batch`、`loss.backward()` 和 `optimizer.step()` 这一整套流程，让我对反向传播和梯度下降有了更直观的认识。最后那个动态调整学习率的实践，也让我真实体会到“调参”对模型训练结果有多重要。



### 2. 实验 4.2 感想

实验 4.2 的“以图搜图”效果非常好，这让我直观体会到了预训练模型的实用价值。

相比于 Lab3 的颜色直方图，ResNet50 提取的 2048 维特征包含了丰富的高级语义信息，这直接体现在了检索的高准确率上。这也让我明白了“迁移学习”为什么是现在的主流：我们不需要从零开始训练，就能利用 ImageNet 上学到的知识来解决我们自己的检索任务。

在具体的实现上，这个实验也让我学到了很多“工程”上的技巧：

1. 特征截获：学会了如何通过重写 `features` 函数来“截获”模型在 `avgpool` 层的输出，拿到了我们真正需要的特征向量，而不是最终的分类结果。
2. 离线/在线分离：`build_database` 函数的设计思路很实用。它把计算量大、耗时长的特征提取（N 次 CNN 传播）作为“离线”步骤一次性完成，而把计算量小、速度快的检索（N 次向量点积）作为“在线”步骤。在实际应用中，这种“预处理”和“实时查询”分离的思想非常重要。
3. 系统实现：我认识到一个完整的应用不只有核心算法。我们还需要用 `os.walk` 来灵活地处理文件目录结构，用 `matplotlib` 来生成直观的可视化报告，这些都是构成一个健壮程序不可或缺的部分。

---



## 六、源程序

### `lab4_1.py`

```python
# SJTU EE208

'''Train CIFAR-10 with PyTorch.'''
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import resnet20

start_epoch = 0
end_epoch = 4
lr = 0.1

# Data pre-processing, DO NOT MODIFY
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

classes = ("airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck")

# Model
print('==> Building model..')
model = resnet20()
# If you want to restore training (instead of training from beginning),
# you can continue training based on previously-saved models
# by uncommenting the following two lines.
# Do not forget to modify start_epoch and end_epoch.
# restore_model_path = 'pretrained/ckpt_4_acc_63.320000.pth'
# model.load_state_dict(torch.load(restore_model_path)['net'])

# A better method to calculate loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        # The outputs are of size [128x10].
        # 128 is the number of images fed into the model 
        # (yes, we feed a certain number of images into the model at the same time, 
        # instead of one by one)
        # For each image, its output is of length 10.
        # Index i of the highest number suggests that the prediction is classes[i].
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('Epoch [%d] Batch [%d/%d] Loss: %.3f | Traininig Acc: %.3f%% (%d/%d)'
              % (epoch, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1),
                 100. * correct / total, correct, total))


def test(epoch):
    print('==> Testing...')
    model.eval()

    test_loss = 0 
    correct = 0   
    total = 0     

    with torch.no_grad():
        ##### TODO: calc the test accuracy #####
        # Hint: You do not have to update model parameters.
        #       Just get the outputs and count the correct predictions.
        #       You can turn to `train` function for help.
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total if total > 0 else 0
        ########################################
    # Save checkpoint.
    print('Test Acc: %f' % acc)
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        
    torch.save(state, './checkpoint/ckpt_%d_acc_%f.pth' % (epoch,acc))

    if epoch == 9:
        torch.save(state, './checkpoint/final.pth')
        

print('==> Starting training for 10 epochs...')

for epoch in range(0, 10):
    if epoch == 5:
        print('==> Adjusting learning rate to 0.01')
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
    
    train(epoch)
    test(epoch)

print('==> Training finished.')
```

### `lab4_2.py` 

```python
import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision.models import resnet50, ResNet50_Weights 


LIBRARY_DIR = 'image_library'       # 图像库
QUERY_DIR = 'query'                 # 待搜索图片
OUTPUT_DIR = 'output'               # 保存结果图
TOP_K = 5                           # Top 5 结果 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('Load model: ResNet50')

weights = ResNet50_Weights.DEFAULT 
model = resnet50(weights=weights)

model.eval()  
model.to(device)

# 图像预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# 图片加载和显示预处理
display_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
])

# 特征提取函数
def features(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x

def extract_feature(image_path, model, preprocess):
    try:
        img = default_loader(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feature = features(input_batch)
    feature_np = feature.cpu().detach().numpy().flatten()
    feature_norm = feature_np / np.linalg.norm(feature_np)
    return feature_norm


# 构建特征数据库
def build_database(library_dir, model, preprocess):
    db_features = []
    db_image_paths = []
    print(f"\nBuilding feature database for '{library_dir}' (scanning all sub-directories)...")
    for dirpath, dirnames, filenames in os.walk(library_dir):
        image_files = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in sorted(image_files):
            image_path = os.path.join(dirpath, filename)
            feature = extract_feature(image_path, model, preprocess)
            if feature is not None:
                db_features.append(feature)
                db_image_paths.append(image_path)
    
    total_features = len(db_features)
    print(f"Database build complete. Total features extracted: {total_features}")
    return np.array(db_features), db_image_paths


# 绘图函数
def plot_results(query_path, query_filename, matched_paths, scores, top_k):
    if not matched_paths:
        print("No matched images to plot.")
        return

    fig, axes = plt.subplots(1, 6, figsize=(24, 5)) 
    fig.suptitle(f"Query: {os.path.basename(query_path)} - Top {top_k} Matches", fontsize=16)

    # 显示查询图片
    try:
        img_query = Image.open(query_path).convert('RGB')
        img_query = display_transform(img_query) 
        
        axes[0].imshow(img_query)
        axes[0].set_title(f"Query\n({os.path.basename(query_path)})", fontsize=10)
        axes[0].axis('off')
    except Exception as e:
        axes[0].set_title(f"Error loading query\n{os.path.basename(query_path)}\n({e})", fontsize=8)
        axes[0].axis('off')

    # 显示匹配图片
    for i in range(top_k):
        if i < len(matched_paths):
            ax_current = axes[i+1] 
            
            try:
                img_match = Image.open(matched_paths[i]).convert('RGB')
                img_match = display_transform(img_match) 
                ax_current.imshow(img_match)
                
                relative_path = os.path.relpath(matched_paths[i], LIBRARY_DIR)
                ax_current.set_title(f"Rank {i+1} (Score: {scores[i]:.4f})\n({relative_path})", fontsize=10)
                
                ax_current.axis('off')
            except Exception as e:
                ax_current.set_title(f"Error loading match {i+1}\n({os.path.basename(matched_paths[i])})\n({e})", fontsize=8)
                ax_current.axis('off')
        else:
            axes[i+1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    base_filename = os.path.splitext(query_filename)[0]
    output_filename = f"{base_filename}_result.png"
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    
    plt.savefig(save_path)
    print(f"Result plot saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    
    # 构建特征数据库
    db_features, db_paths = build_database(LIBRARY_DIR, model, trans)
    
    if db_features.shape[0] == 0:
        print(f"Error: No features were extracted. Is '{LIBRARY_DIR}' empty or incorrect?")
    
    elif not os.path.exists(QUERY_DIR):
        print(f"Error: Query directory not found at '{QUERY_DIR}'")
        
    else:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory at: {OUTPUT_DIR}")

        # 遍历 query 文件夹中的所有图片
        query_image_files = sorted([f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not query_image_files:
            print(f"Error: No query images found in '{QUERY_DIR}'")
        
        for query_filename in query_image_files:
            query_image_path = os.path.join(QUERY_DIR, query_filename)
            
            print(f"\n======== Processing Query Image: {query_image_path} ========")
            
            # 提取查询图片的特征
            query_feature = extract_feature(query_image_path, model, trans)
            
            if query_feature is None:
                print(f"Skipping {query_image_path} due to extraction error.")
                continue 
            
            # 计算相似度
            scores = np.dot(db_features, query_feature)
            
            # 排序并报告
            top_indices = np.argsort(scores)[::-1][:TOP_K]
            
            matched_image_paths = [db_paths[idx] for idx in top_indices]
            matched_scores = [scores[idx] for idx in top_indices]

            print(f"\n--- Top {TOP_K} Search Results for '{query_image_path}' ---")
            print("Similarity Method: Cosine Similarity (Vector Angle)")
            
            for i in range(len(matched_image_paths)):
                print(f"\nRank {i+1}:")
                print(f"  Image: {matched_image_paths[i]}")
                print(f"  Score: {matched_scores[i]:.6f}")
            
            # 绘制并保存结果图
            plot_results(query_image_path, query_filename, matched_image_paths, matched_scores, TOP_K)

            print(f"======== Finished Processing: {query_image_path} ========\n")
```

### `download.py`

```python
import os
from bing_image_downloader import downloader

def download_images_task():
    """
    下载图片,建立图像库 (Library)
    """
    # 图像库的主目录
    library_path = 'image_library'
    if not os.path.exists(library_path):
        os.makedirs(library_path)
        print(f"创建目录: {library_path}")
    else:
        print(f"目录 '{library_path}' 已存在。")

    queries = [
        "panda",      
        "cat",         
        "dog",          
        "airplane",     
        "car",          
        "bicycle",     
        "flower",      
        "beach",        
        "building",    
        "tree"         
    ]
    
    # 每个类别下载8张
    limit_per_query = 8

    print(f"--- 开始下载图片到 '{library_path}' 目录 ---")
    print(f"总共 {len(queries)} 个类别, 每个类别最多 {limit_per_query} 张。")

    for query in queries:
        print(f"正在下载 '{query}' ...")
        downloader.download(
            query,
            limit=limit_per_query,
            output_dir=library_path,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )
    
    print("--- 所有图片下载完成 ---")
    print(f"请检查 '{library_path}' 文件夹。")

# --- 主程序入口 ---
if __name__ == "__main__":
    download_images_task()
```
