## base

- 50%概率竖直轴镜像
- 50%概率绕特征点旋转（0均值10标准差的高斯分布，单位度）
- 训练样本窗口大小为特征点正方形包围盒四个方向各扩1/6倍，此基础上增加抖动（-0.01~0.01的均匀分布，单位原始正方形包围盒长度）

## pad

- 50%概率竖直轴镜像
- 50%概率绕特征点旋转（0均值10标准差的高斯分布，单位度）
- 训练样本窗口大小为特征点正方形包围盒四个方向各扩1/6倍，此基础上增加抖动（-1/6~1/6的均匀分布，单位原始正方形包围盒长度）

## color

- 50%概率竖直轴镜像
- 50%概率绕特征点旋转（0均值10标准差的高斯分布，单位度）
- 训练样本窗口大小为特征点正方形包围盒四个方向各扩1/6倍，此基础上增加抖动（-0.01~0.01的均匀分布，单位原始正方形包围盒长度）
- 输入图片初始像素值范围缩放至-1~1

## augment-1

- 50%概率竖直轴镜像
- 50%概率绕特征点旋转（0均值10标准差的高斯分布，单位度）
- 特征点包围盒平移扰动（-1/6~1/6的均匀分布，单位原始正方形包围盒长度）
- 训练样本窗口大小为特征点正方形包围盒四个方向各扩1/6倍，此基础上增加抖动（-0.01~0.01的均匀分布，单位原始正方形包围盒长度）

## augment-2

- 50%概率竖直轴镜像
- 50%概率绕特征点旋转（0均值10标准差的高斯分布，单位度）
- 特征点包围盒平移扰动（-1/12~1/12的均匀分布，单位原始正方形包围盒长度）
- 训练样本窗口大小为特征点正方形包围盒四个方向各扩1/6倍，此基础上增加抖动（-0.01~0.01的均匀分布，单位原始正方形包围盒长度）
