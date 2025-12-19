import pandas as pd
import matplotlib.pyplot as plt
import os
from tkinter import filedialog

# 导入新的文献分析模块（负责PDF文献解析、文本分析、词云生成等功能）
from financial_document_analyzer import FinancialDocumentAnalyzer


class Experiment1:
    """
    实验一核心业务类
    功能：整合金融文献分析和金融数据文件分析两大核心功能，
          实现数据加载、分析、可视化、结果保存全流程
    """

    def __init__(self, visualizer):
        """
        初始化实验一模块
        参数：
            visualizer: 可视化工具实例 - 提供图表绘制和保存功能
        """
        self.visualizer = visualizer  # 可视化工具实例
        self.current_data = None  # 当前加载的金融数据（DataFrame）
        self.current_file_path = None  # 当前加载文件的路径
        self.analysis_results = None  # 数据分析结果缓存

        # 初始化文献分析器实例（处理PDF文献解析、文本分析等）
        self.document_analyzer = FinancialDocumentAnalyzer()

    # ==================== 文献分析相关方法 ====================
    def load_document(self, log_callback=None):
        """
        加载金融文献（PDF格式）
        参数：
            log_callback: function - 日志回调函数，用于输出加载过程信息
        返回：
            bool - 加载成功返回True，失败返回False
        """
        return self.document_analyzer.load_document(log_callback)

    def analyze_document(self, log_callback=None):
        """
        分析金融文献内容
        功能：提取文本、中文分词、词频统计、情感分析
        参数：
            log_callback: function - 日志回调函数，输出分析过程和结果
        返回：
            bool - 分析成功返回True，失败返回False
        """
        return self.document_analyzer.analyze_document(log_callback)

    def visualize_document(self, log_callback=None):
        """
        可视化文献分析结果
        功能：生成词云图、词频柱状图、情感分析图等
        参数：
            log_callback: function - 日志回调函数，输出可视化过程信息
        返回：
            bool - 可视化成功返回True，失败返回False
        """
        return self.document_analyzer.visualize_analysis(log_callback)

    def save_document_analysis(self, log_callback=None):
        """
        保存文献分析结果
        功能：保存分析报告、词云图片、词频数据等文件
        参数：
            log_callback: function - 日志回调函数，输出保存过程信息
        返回：
            bool - 保存成功返回True，失败返回False
        """
        return self.document_analyzer.save_analysis_results(log_callback)

    def clear_document_data(self, log_callback=None):
        """
        重置文献数据
        功能：清空已加载的文献内容和分析结果
        参数：
            log_callback: function - 日志回调函数，输出重置状态
        返回：
            bool - 重置成功返回True，失败返回False
        """
        return self.document_analyzer.clear_document_data(log_callback)

    # ==================== 数据分析相关方法 ====================
    def load_data_file(self, log_callback=None):
        """
        加载已有的金融数据文件（CSV/Excel格式）
        参数：
            log_callback: function - 日志回调函数，输出加载过程和结果
        返回：
            bool - 加载成功返回True，失败返回False
        """
        try:
            # 弹出文件选择对话框，限定文件类型
            file_path = filedialog.askopenfilename(
                title="选择金融数据文件",
                filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
            )

            # 用户取消选择文件
            if not file_path:
                return False

            # 输出加载文件信息
            if log_callback:
                log_callback(f"正在加载数据文件: {os.path.basename(file_path)}")

            # 根据文件后缀选择读取方式
            if file_path.endswith('.csv'):
                # 读取CSV文件，使用utf-8-sig编码处理中文
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_path.endswith('.xlsx'):
                # 读取Excel文件
                df = pd.read_excel(file_path)
            else:
                # 不支持的文件格式
                if log_callback:
                    log_callback("不支持的文件格式")
                return False

            # 保存加载的数据和文件路径
            self.current_data = df
            self.current_file_path = file_path

            # 输出加载成功信息和数据预览
            if log_callback:
                log_callback(f"数据加载成功，共 {len(df)} 条记录")
                log_callback("\n数据预览:")
                log_callback(df.head().to_string())  # 显示前5行数据

            return True

        except Exception as e:
            # 捕获并输出加载异常
            if log_callback:
                log_callback(f"加载数据文件失败: {str(e)}")
            return False

    def analyze_existing_data(self, log_callback=None):
        """
        分析已有数据文件（仅执行分析，不显示图表）
        功能：基础统计分析、缺失值检测、数据类型识别
        参数：
            log_callback: function - 日志回调函数，输出分析过程和结果
        返回：
            bool - 分析成功返回True，失败返回False
        """
        # 数据校验：检查是否已加载数据
        if self.current_data is None:
            if log_callback:
                log_callback("请先加载数据文件")
            return False

        try:
            if log_callback:
                log_callback("开始数据分析...")

            # 执行基础统计分析，保存结果到实例变量
            self.analysis_results = self.basic_data_analysis(log_callback)

            if log_callback:
                log_callback("数据分析完成！")

            return True

        except Exception as e:
            # 捕获并输出分析异常
            if log_callback:
                log_callback(f"数据分析失败: {str(e)}")
            return False

    def visualize_data(self, log_callback=None):
        """
        可视化金融数据（仅显示图表，不保存文件）
        功能：生成股价走势图、成交量柱状图、涨跌幅分布图等
        参数：
            log_callback: function - 日志回调函数，输出可视化过程信息
        返回：
            bool - 可视化成功返回True，失败返回False
        """
        # 数据校验：检查是否已加载数据
        if self.current_data is None:
            if log_callback:
                log_callback("请先加载数据文件")
            return False

        try:
            df = self.current_data

            # 检查必要的数据列（收盘价、成交量、涨跌幅）
            required_columns = ['close', 'vol', 'pct_chg']
            missing_columns = [col for col in required_columns if col not in df.columns]

            # 缺少必要列则终止可视化
            if missing_columns:
                if log_callback:
                    log_callback(f"缺少必要的列: {missing_columns}")
                    log_callback("请确保数据包含: close(收盘价), vol(成交量), pct_chg(涨跌幅) 列")
                return False

            if log_callback:
                log_callback("正在生成可视化图表...")

            # 提取文件名（不含后缀）作为图表标题前缀
            file_name = os.path.basename(self.current_file_path).split('.')[0]
            # 调用可视化工具绘制基础图表
            success = self.visualizer.plot_simple_charts(df, file_name)

            # 输出可视化完成信息
            if success and log_callback:
                log_callback("数据可视化完成！图表已显示")

            return success

        except Exception as e:
            # 捕获并输出可视化异常
            if log_callback:
                log_callback(f"数据可视化失败: {str(e)}")
            return False

    def save_analysis_results(self, log_callback=None):
        """
        保存数据分析结果（包括文本报告、处理后数据、图表文件）
        参数：
            log_callback: function - 日志回调函数，输出保存过程信息
        返回：
            bool - 保存成功返回True，失败返回False
        """
        # 数据校验：检查是否已加载数据
        if self.current_data is None:
            if log_callback:
                log_callback("请先加载数据文件")
            return False

        # 校验：检查是否已完成分析
        if self.analysis_results is None:
            if log_callback:
                log_callback("请先进行数据分析")
            return False

        try:
            if log_callback:
                log_callback("正在保存分析结果...")

            # 创建结果保存目录（不存在则新建）
            result_dir = "result/数据分析结果"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                log_callback(f"创建目录: {result_dir}")

            # 提取文件名（不含后缀）
            file_name = os.path.basename(self.current_file_path).split('.')[0]

            # 1. 保存分析报告（文本文件）
            report_content = self.generate_analysis_report()
            report_path = os.path.join(result_dir, f"{file_name}_分析报告.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            # 2. 保存处理后的数据（CSV格式）
            data_path = os.path.join(result_dir, f"{file_name}_处理数据.csv")
            self.current_data.to_csv(data_path, index=False, encoding='utf-8-sig')

            # 3. 保存所有可视化图表
            chart_success = self.visualizer.save_all_charts(self.current_data, file_name)

            # 输出保存成功信息
            if log_callback:
                log_callback(f"分析结果已保存到: {result_dir}")
                log_callback(f"生成的文件包括:")
                log_callback(f"- {file_name}_分析报告.txt（分析报告）")
                log_callback(f"- {file_name}_处理数据.csv（处理后的数据）")
                if chart_success:
                    log_callback(f"- 多种分析图表文件（走势图、成交量图等）")

            return True

        except Exception as e:
            # 捕获并输出保存异常
            if log_callback:
                log_callback(f"保存分析结果失败: {str(e)}")
            return False

    def basic_data_analysis(self, log_callback=None):
        """
        执行基础数据分析（核心分析方法）
        功能：统计记录数、识别数据类型、检测缺失值、计算数值列统计指标
        参数：
            log_callback: function - 日志回调函数，输出分析结果
        返回：
            dict - 包含所有分析结果的字典
        """
        df = self.current_data

        # 构建分析结果字典
        analysis = {
            'total_records': len(df),  # 总记录数
            'columns': list(df.columns),  # 数据列名列表
            'data_types': df.dtypes.to_dict(),  # 各列数据类型
            'missing_values': df.isnull().sum().to_dict(),  # 各列缺失值数量
            'basic_stats': {},  # 数值列统计指标
            'file_name': os.path.basename(self.current_file_path)  # 文件名
        }

        # 筛选数值列（整数/浮点数），计算统计指标
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            analysis['basic_stats'][col] = {
                'mean': df[col].mean(),  # 平均值
                'std': df[col].std(),  # 标准差
                'min': df[col].min(),  # 最小值
                'max': df[col].max(),  # 最大值
                'median': df[col].median()  # 中位数
            }

        # 输出分析结果到日志
        if log_callback:
            log_callback("\n=== 基本统计分析结果 ===")
            log_callback(f"总记录数: {analysis['total_records']}")
            log_callback(f"数据列: {', '.join(analysis['columns'])}")
            log_callback("\n缺失值统计:")
            # 仅输出有缺失值的列
            for col, missing in analysis['missing_values'].items():
                if missing > 0:
                    log_callback(f"  {col}: {missing} 个缺失值")

            # 输出数值列统计信息
            log_callback("\n数值列统计信息:")
            for col, stats in analysis['basic_stats'].items():
                log_callback(f"\n{col}:")
                log_callback(f"  平均值: {stats['mean']:.4f}")  # 保留4位小数
                log_callback(f"  标准差: {stats['std']:.4f}")
                log_callback(f"  最小值: {stats['min']:.4f}")
                log_callback(f"  最大值: {stats['max']:.4f}")
                log_callback(f"  中位数: {stats['median']:.4f}")

        return analysis

    def generate_analysis_report(self):
        """
        生成格式化的数据分析报告（文本内容）
        返回：
            str - 完整的分析报告文本
        """
        # 构建报告标题和基础信息
        report_content = f"金融数据分析报告\n"
        report_content += f"=" * 50 + "\n"
        report_content += f"分析文件: {self.analysis_results['file_name']}\n"
        report_content += f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"总记录数: {self.analysis_results['total_records']}\n"
        report_content += f"数据列: {', '.join(self.analysis_results['columns'])}\n\n"

        # 添加缺失值统计
        report_content += "缺失值统计:\n"
        for col, missing in self.analysis_results['missing_values'].items():
            report_content += f"  {col}: {missing} 个缺失值\n"

        # 添加数值列统计信息
        report_content += "\n基本统计信息:\n"
        for col, stats in self.analysis_results['basic_stats'].items():
            report_content += f"\n{col}:\n"
            report_content += f"  平均值: {stats['mean']:.4f}\n"
            report_content += f"  标准差: {stats['std']:.4f}\n"
            report_content += f"  最小值: {stats['min']:.4f}\n"
            report_content += f"  最大值: {stats['max']:.4f}\n"
            report_content += f"  中位数: {stats['median']:.4f}\n"

        return report_content

    def clear_data(self, log_callback=None):
        """
        重置所有金融数据相关状态
        功能：清空已加载的数据、文件路径、分析结果
        参数：
            log_callback: function - 日志回调函数，输出重置状态
        """
        self.current_data = None  # 清空当前数据
        self.current_file_path = None  # 清空文件路径
        self.analysis_results = None  # 清空分析结果
        if log_callback:
            log_callback("数据已重置：已清空加载的文件和分析结果")
