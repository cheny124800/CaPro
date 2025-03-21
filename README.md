# CaPro

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
python .\FreeCOS-main\Data\CrackTree\convert_gray.py
```

> [!NOTE]
>
> 如需对生成gt颜色（黑白交换等），可在 convert_gt line 47/48 进行更改

曲线结构对应的box的txt文件在 .\FreeCOS-main\Data\CrackTree\txt_data 目录下

