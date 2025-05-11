#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立的TensorBoard启动脚本，用于启动TensorBoard服务器查看训练日志
"""

import os
import sys
import argparse
import logging

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入TensorBoard工具模块
from src.utils.tensorboard_utils import start_tensorboard, check_tensorboard_running

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TensorBoard启动工具 - 用于查看训练日志",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="TensorBoard日志目录路径"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006,
        help="TensorBoard服务器端口"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="TensorBoard服务器主机地址"
    )
    parser.add_argument(
        "--background", 
        action="store_true",
        help="在后台运行TensorBoard服务器"
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="仅检查指定端口是否有TensorBoard正在运行"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 如果只是检查TensorBoard状态
    if args.check:
        is_running = check_tensorboard_running(args.port, args.host)
        if is_running:
            logger.info(f"TensorBoard正在端口 {args.port} 运行")
            logger.info(f"您可以通过访问 http://{args.host}:{args.port} 查看TensorBoard")
        else:
            logger.info(f"端口 {args.port} 上没有运行TensorBoard")
        return 0
    
    # 确保日志目录存在
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.exists(log_dir):
        logger.warning(f"TensorBoard日志目录 {log_dir} 不存在，将创建该目录")
        os.makedirs(log_dir, exist_ok=True)
    
    # 启动TensorBoard
    try:
        logger.info(f"正在启动TensorBoard，日志目录：{log_dir}")
        process = start_tensorboard(
            log_dir=log_dir,
            port=args.port,
            host=args.host,
            background=args.background
        )
        
        if args.background and process:
            logger.info(f"TensorBoard已在后台启动，可通过 http://{args.host}:{args.port} 访问")
            logger.info("要停止TensorBoard，请终止此Python进程")
            # 保持脚本运行，直到用户手动停止
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("接收到中断信号，正在停止TensorBoard服务器...")
        else:
            logger.info("TensorBoard已退出")
        
        return 0
    except Exception as e:
        logger.error(f"启动TensorBoard时出错：{e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 