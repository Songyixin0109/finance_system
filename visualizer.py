import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import MonthLocator, DateFormatter
import os

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """金融数据可视化类：提供股票分析图表绘制与保存功能"""

    def __init__(self):
        self.figsize = (15, 10)  # 默认图表尺寸

    def plot_stock_analysis(self, df, stock_name):
        """绘制股票分析组合图表（2×2布局）"""
        try:
            # 确保日期为datetime格式
            df = df.copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # 创建2×2子图布局
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle(f'{stock_name} 股票分析', fontsize=16)

            # 绘制各子图
            self._plot_price_trend(axes[0, 0], df, '收盘价走势图')
            self._plot_volume_chart(axes[0, 1], df, '成交量柱状图')
            self._plot_returns_histogram(axes[1, 0], df, '涨跌幅分布直方图')
            self._plot_price_simulation(axes[1, 1], df, '价格波动模拟图')

            plt.tight_layout()
            plt.show()
            return True

        except Exception as e:
            print(f"绘图错误: {str(e)}")
            return False

    def _plot_price_trend(self, ax, df, title):
        """绘制收盘价走势子图"""
        ax.plot(df['trade_date'], df['close'], label='收盘价', color='red', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('日期')
        ax.set_ylabel('价格(元)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 设置X轴按月显示
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_volume_chart(self, ax, df, title):
        """绘制成交量柱状子图（按周聚合）"""
        # 按周聚合成交量，提升可读性
        weekly_volume = df.set_index('trade_date')['vol'].resample('W').sum()

        ax.bar(weekly_volume.index, weekly_volume.values, alpha=0.7, color='blue', width=5)
        ax.set_title(title)
        ax.set_xlabel('日期')
        ax.set_ylabel('成交量')
        ax.grid(True, alpha=0.3)

        # 设置X轴按月显示
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_returns_histogram(self, ax, df, title):
        """绘制涨跌幅分布直方图"""
        returns = df['pct_chg']
        ax.hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('涨跌幅(%)')
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)

        # 添加均值和标准差参考线
        mean_return = returns.mean()
        std_return = returns.std()
        ax.axvline(mean_return, color='red', linestyle='--', label=f'均值: {mean_return:.2f}%')
        ax.axvline(mean_return + std_return, color='orange', linestyle='--', alpha=0.7, label=f'±1标准差')
        ax.axvline(mean_return - std_return, color='orange', linestyle='--', alpha=0.7)
        ax.legend()

    def _plot_price_simulation(self, ax, df, title):
        """绘制价格波动模拟子图（最近50个交易日）"""
        # 取最近50个交易日数据，避免图表拥挤
        display_data = df.tail(50)
        dates = range(len(display_data))

        ax.plot(dates, display_data['open'], label='开盘价', color='blue', linewidth=1, alpha=0.7)
        ax.plot(dates, display_data['high'], label='最高价', color='red', linewidth=1, alpha=0.7)
        ax.plot(dates, display_data['low'], label='最低价', color='green', linewidth=1, alpha=0.7)
        ax.plot(dates, display_data['close'], label='收盘价', color='black', linewidth=2)

        ax.set_title(title)
        ax.set_xlabel('最近50个交易日')
        ax.set_ylabel('价格(元)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_prediction_results(self, actual, predictions, algorithm_names):
        """绘制机器学习预测结果对比图"""
        try:
            plt.figure(figsize=(12, 6))

            x = range(len(actual))
            plt.plot(x, actual, label='实际值', linewidth=3, color='black', marker='o', markersize=4)

            # 为不同算法分配颜色和标记
            colors = ['red', 'blue', 'green']
            markers = ['s', '^', 'D']

            for i, (name, pred) in enumerate(zip(algorithm_names, predictions)):
                plt.plot(x, pred, label=name, linestyle='--', alpha=0.8,
                         color=colors[i], marker=markers[i], markersize=4)

            plt.title('机器学习预测结果对比')
            plt.xlabel('测试样本')
            plt.ylabel('股价')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            return True
        except Exception as e:
            print(f"预测结果绘图错误: {str(e)}")
            return False

    def plot_simple_charts(self, df, stock_name):
        """快捷调用组合图表绘制方法"""
        try:
            success = self.plot_stock_analysis(df, stock_name)
            return success
        except Exception as e:
            print(f"显示图表错误: {str(e)}")
            return False

    def save_all_charts(self, df, stock_name):
        """保存所有股票分析图表到指定目录（不显示）"""
        try:
            # 关闭交互模式，避免GUI警告
            plt.ioff()

            # 创建结果保存目录
            base_dir = "result"
            analysis_dir = os.path.join(base_dir, "数据分析结果")
            if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)

            # 确保日期格式正确
            df = df.copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # 1. 保存收盘价走势图
            fig1 = plt.figure(figsize=(12, 6))
            plt.plot(df['trade_date'], df['close'], color='red', linewidth=2)
            plt.title(f'{stock_name} - 收盘价走势')
            plt.xlabel('日期')
            plt.ylabel('价格(元)')
            plt.gca().xaxis.set_major_locator(MonthLocator(bymonthday=1))
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            file_path1 = os.path.join(analysis_dir, f'{stock_name}_收盘价走势.png')
            plt.savefig(file_path1, dpi=300, bbox_inches='tight')
            plt.close(fig1)

            # 2. 保存成交量图
            fig2 = plt.figure(figsize=(12, 6))
            weekly_volume = df.set_index('trade_date')['vol'].resample('W').sum()
            plt.bar(weekly_volume.index, weekly_volume.values, alpha=0.7, color='blue', width=5)
            plt.title(f'{stock_name} - 成交量')
            plt.xlabel('日期')
            plt.ylabel('成交量')
            plt.gca().xaxis.set_major_locator(MonthLocator(bymonthday=1))
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            file_path2 = os.path.join(analysis_dir, f'{stock_name}_成交量.png')
            plt.savefig(file_path2, dpi=300, bbox_inches='tight')
            plt.close(fig2)

            # 3. 保存涨跌幅分布图
            fig3 = plt.figure(figsize=(10, 6))
            returns = df['pct_chg']
            plt.hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.title(f'{stock_name} - 涨跌幅分布')
            plt.xlabel('涨跌幅(%)')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            file_path3 = os.path.join(analysis_dir, f'{stock_name}_涨跌幅分布.png')
            plt.savefig(file_path3, dpi=300, bbox_inches='tight')
            plt.close(fig3)

            # 4. 保存价格波动图
            fig4 = plt.figure(figsize=(12, 6))
            display_data = df.tail(50)
            dates = range(len(display_data))
            plt.plot(dates, display_data['open'], label='开盘价', color='blue', linewidth=1, alpha=0.7)
            plt.plot(dates, display_data['high'], label='最高价', color='red', linewidth=1, alpha=0.7)
            plt.plot(dates, display_data['low'], label='最低价', color='green', linewidth=1, alpha=0.7)
            plt.plot(dates, display_data['close'], label='收盘价', color='black', linewidth=2)
            plt.title(f'{stock_name} - 价格波动')
            plt.xlabel('最近50个交易日')
            plt.ylabel('价格(元)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            file_path4 = os.path.join(analysis_dir, f'{stock_name}_价格波动.png')
            plt.savefig(file_path4, dpi=300, bbox_inches='tight')
            plt.close(fig4)

            # 5. 保存组合分析图
            fig5, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig5.suptitle(f'{stock_name} 股票分析', fontsize=16)
            self._plot_price_trend(axes[0, 0], df, '收盘价走势图')
            self._plot_volume_chart(axes[0, 1], df, '成交量柱状图')
            self._plot_returns_histogram(axes[1, 0], df, '涨跌幅分布直方图')
            self._plot_price_simulation(axes[1, 1], df, '价格波动模拟图')
            plt.tight_layout()
            file_path5 = os.path.join(analysis_dir, f'{stock_name}_组合分析图.png')
            plt.savefig(file_path5, dpi=300, bbox_inches='tight')
            plt.close(fig5)

            # 恢复交互模式
            plt.ion()
            return True

        except Exception as e:
            print(f"保存图表错误: {str(e)}")
            plt.ion()
            return False

    def save_prediction_chart(self, actual, predictions, algorithm_names, stock_name):
        """保存机器学习预测结果对比图"""
        try:
            plt.ioff()  # 关闭交互模式

            # 创建保存目录
            base_dir = "result"
            analysis_dir = os.path.join(base_dir, "数据分析结果")
            if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)

            fig = plt.figure(figsize=(12, 6))
            x = range(len(actual))
            plt.plot(x, actual, label='实际值', linewidth=3, color='black', marker='o', markersize=4)

            # 为不同算法分配颜色和标记
            colors = ['red', 'blue', 'green']
            markers = ['s', '^', 'D']
            for i, (name, pred) in enumerate(zip(algorithm_names, predictions)):
                plt.plot(x, pred, label=name, linestyle='--', alpha=0.8,
                         color=colors[i], marker=markers[i], markersize=4)

            plt.title(f'{stock_name} - 机器学习预测结果对比')
            plt.xlabel('测试样本')
            plt.ylabel('股价')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 保存图表
            file_path = os.path.join(analysis_dir, f'{stock_name}_预测结果.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            plt.ion()  # 恢复交互模式
            print(f"预测图表已保存: {file_path}")
            return True
        except Exception as e:
            print(f"保存预测图表错误: {str(e)}")
            plt.ion()
            return False