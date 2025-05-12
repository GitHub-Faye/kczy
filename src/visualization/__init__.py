"""
可视化模块，提供各种可视化功能。
"""

# 导出指标绘图功能
from .metrics_plots import (
    save_plot,
    plot_loss,
    plot_accuracy,
    plot_training_history
)

# 导出注意力可视化功能
from .attention_viz import (
    plot_attention_weights,
    visualize_attention_on_image,
    visualize_all_heads
)

# 导出模型结构可视化功能
from .model_viz import (
    plot_model_structure,
    plot_encoder_block,
    visualize_layer_weights
)

# 导出静态可视化功能
from .static_viz import (
    create_model_overview,
    create_attention_analysis,
    create_comprehensive_visualization,
    compare_models,
    generate_visualization_report
)

__all__ = [
    # 指标绘图
    'save_plot',
    'plot_loss',
    'plot_accuracy',
    'plot_training_history',
    
    # 注意力可视化
    'plot_attention_weights',
    'visualize_attention_on_image',
    'visualize_all_heads',
    
    # 模型结构可视化
    'plot_model_structure',
    'plot_encoder_block',
    'visualize_layer_weights',
    
    # 静态可视化
    'create_model_overview',
    'create_attention_analysis',
    'create_comprehensive_visualization',
    'compare_models',
    'generate_visualization_report'
] 