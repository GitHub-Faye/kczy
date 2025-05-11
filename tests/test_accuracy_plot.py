"""
测试准确率绘图函数
"""
import os
import logging

from src.visualization.metrics_plots import plot_accuracy

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 输出目录
plots_dir = os.path.join('temp_metrics', 'plots')
os.makedirs(plots_dir, exist_ok=True)

# 测试准确率绘图函数
print("测试plot_accuracy函数...")
plot_accuracy(
    metrics_path=os.path.join('temp_metrics', 'simulation_train_metrics.csv'),
    output_path=os.path.join(plots_dir, 'accuracy_curve_test.png'),
    experiment_name='Test Run',
    figsize=(8, 4),  # 减小图表尺寸
    dpi=100,         # 降低DPI
    style=None       # 不使用自定义样式
)
print("绘图完成。请检查文件是否生成。") 