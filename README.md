# CaPro

【代码结构和部分内容暂未经优化，不影响结果】

## 步骤一：FreeCOS生成数据

-环境要求：无特别要求

1、生成曲线结构

```python
python .\FreeCOS-main\Data\CrackTree\make_fakevessel.py
```

直接生成的结果会在 .\FreeCOS-main\Data\CrackTree\fake_rgbvessel_thin 目录下

运行以下代码得到 fake_grayvessel_thin 或  fake_gtvessel_thin

```python
# 生成gray版曲线结构
python .\FreeCOS-main\Data\CrackTree\convert_gray.py
# 生成gt版曲线结构
python .\FreeCOS-main\Data\CrackTree\convert_gt.py
```

> [!NOTE]
>
> 如需对生成gt颜色（黑白交换等），可在 convert_gt line 47/48 进行更改，更保守的做法是运行 **.\FreeCOS-main\FreeCOS-main\FDA_RGB\inverse_black_white** 

曲线结构对应的box的txt文件在 .\FreeCOS-main\Data\CrackTree\txt_data 目录下

2、曲线结构嵌入

修改 **.\FreeCOS-main\FDA_RGB> .\FDA_retinal.py** 中的line154 ~ line156，其中tar是背景，src是需要嵌入的曲线结构，随后运行，得到仿真数据集的 <u>**img、mask、txt**</u>，保存目录见line157 ~ line158。

```python
python .\FreeCOS-main\FDA_RGB\FDA_retinal.py
```

## 步骤二：制作训练数据&训练

-环境要求：参考各工程文件的readme

1、生成xml

进入 **.\Capro\RoLabelImg_Transform-master**，将步骤一生成的img、txt复制到.\Capro\RoLabelImg_Transform-master下的对应目录，参照readme文件，先运行get_list得到文件名参数列表，然后运行txt_to_xml即可得到对应xml

2、xml转json

进入 **.\Capro\R-CenterNet-master_OOO\R-CenterNet-master\labelGenerator**，在voc2coco.py中修改line 68为自己生成的img（曲线结构嵌入图）目录，修改line 212为自己刚刚生成的xml目录，运行代码，结果会保存./data/airplane/annotations/train.json（line 216）

3、训练检测模型

可以直接复制刚刚生成train.json为val.json，把生成的img（曲线结构嵌入图）复制到./data/airplane/images/，运行train.py即可开始训练。

## 步骤三：检测&筛选后分割（SAM）

1、R-CenterNet-master_OOO中得到预测结果

训练完成后运行predict.py可得到预测结果，结果的目录见代码。

2、筛选TopK的检测框

进入**.\Capro\segment-anything-main**（较大需解压），修改TopK_Box.py中line 92 ~ line 95，修改line 105的超参数，运行TopK_Box.py得到txt_selected即为筛选后的txt列表。

3、修改 O_SAM.py中line 96 ~ line 100为自己的目录，<u>值得注意的是</u>，backup_txt_O_dir为备选的txt输入目录，当txt_O_dir中无法找到对应文件（筛选时框不够时会出现此类情况）。修改line 88与line89选择自己需要使用的预训练模型（压缩包中都有）。在line 163可修改GT目录中文件名的读取方式。

4、运行O_SAM.py，得到分割结果与F1指标、MIoU指标。

> [!NOTE]
>
> 一些其他代码的功能：
>
> 在**.\Capro\segment-anything-main**中:
>
> 1、Cal_IOU_F1.py可基于分割结果和GT直接计算指标
>
> 2、ChooseK.py可直接截断txt文件中前K行，因此建议首次运行TopK时可将K设为一个较大的保守值，通过ChooseK截断以节省时间。
>
> 3、预训练模型需要自行下载
>
> 在**.\Capro\R-CenterNet-master_OOO\R-CenterNet-master**中：
>
> 1、bmp2png.py和png2jpg.py用于转格式
>
> 

