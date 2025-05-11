# TensorBoard 使用指南

本文档提供了使用TensorBoard监控和分析模型训练过程的详细说明。

## 启动TensorBoard

有三种方式可以启动TensorBoard服务器：

### 1. 在训练期间启动

在运行训练脚本时添加`--start-tensorboard`参数：

```bash
python scripts/train.py --enable-tensorboard --start-tensorboard
```

其他相关参数：
- `--tensorboard-port 6006`：指定TensorBoard服务器端口（默认为6006）
- `--tensorboard-host localhost`：指定TensorBoard服务器主机（默认为localhost）
- `--tensorboard-background`：在后台运行TensorBoard服务器
- `--tensorboard-dir logs`：指定TensorBoard日志目录（默认为logs）

### 2. 使用专用脚本启动

可以使用单独的脚本启动TensorBoard：

```bash
python scripts/start_tensorboard.py --log-dir logs
```

可选参数：
- `--port 6006`：指定端口（默认为6006）
- `--host localhost`：指定主机（默认为localhost）
- `--background`：在后台运行
- `--check`：仅检查指定端口是否有TensorBoard正在运行

### 3. 直接使用tensorboard命令

如果已安装TensorBoard，可以直接使用命令行工具：

```bash
tensorboard --logdir=logs --port=6006
```

## 访问TensorBoard Web界面

启动TensorBoard服务器后，在浏览器中访问以下URL：

```
http://localhost:6006
```

如果使用了自定义端口或主机，请相应地调整URL。

## TensorBoard界面功能导航

TensorBoard界面分为几个主要标签页，每个标签页显示不同类型的数据：

### 1. SCALARS（标量）

显示随时间变化的指标，如损失值、准确率等。这是最常用的视图，用于监控训练进度。

功能：
- 缩放：拖动鼠标可缩放图表
- 平滑：使用滑块调整曲线平滑度
- 导出：下载CSV格式的数据
- 比较：同时查看多个运行结果

### 2. IMAGES（图像）

显示训练过程中记录的样本图像，如输入样本、中间结果等。

### 3. HISTOGRAMS（直方图）

显示参数分布的直方图，如网络权重和梯度的分布情况。

### 4. DISTRIBUTIONS（分布）

类似于直方图，但提供了更详细的分布视图。

### 5. GRAPHS（图表）

显示模型的计算图，帮助理解模型结构和数据流。

### 6. HPARAMS（超参数）

显示不同超参数配置的实验结果，适用于超参数调优。

### 7. TIME SERIES（时间序列）

显示更详细的时间序列数据。

## 使用技巧

1. **比较多个实验**：在同一图表上查看不同实验的结果
   - 使用不同的日志子目录存储不同的实验
   - 在TensorBoard中可以选择要显示的实验

2. **自定义显示**：
   - 使用正则表达式过滤标签
   - 调整平滑系数以获得更清晰的趋势

3. **导出数据**：
   - 可以将图表导出为CSV格式进行进一步分析
   - 可以将图像导出为PNG文件

4. **实时监控**：
   - TensorBoard会自动刷新，显示最新的训练数据
   - 训练期间可以随时查看进度，无需重启TensorBoard

## 故障排除

1. **TensorBoard无法启动**：
   - 检查端口是否被占用
   - 确认已安装TensorBoard（`pip install tensorboard`）
   - 检查日志目录是否存在

2. **数据不显示**：
   - 确认日志目录路径正确
   - 确认已启用TensorBoard日志记录（`--enable-tensorboard`）
   - 检查训练过程是否正在记录数据

3. **实时更新问题**：
   - 使用浏览器刷新按钮手动刷新
   - 检查是否正确关闭了SummaryWriter（使用`writer.close()`）

## 进阶用法

1. **Profile模式**：
   - 使用`--tensorboard-profile`启用性能分析
   - 帮助识别训练中的性能瓶颈

2. **自定义插件**：
   - TensorBoard支持多种插件扩展功能
   - 可以根据需要添加自定义可视化

3. **远程访问**：
   - 设置`--tensorboard-host 0.0.0.0`允许远程访问
   - 确保配置适当的网络安全措施 