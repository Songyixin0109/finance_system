import threading
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_predictor import CompleteStockAnalyzer, MLVisualizer

# 设置matplotlib中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class DatabaseManager:
    """
    数据库管理类（模拟实现）
    功能：使用字典模拟数据库，实现股票数据的保存和加载
    """

    def __init__(self):
        """初始化数据库管理器，创建空字典作为数据缓存"""
        self.data_cache = {}  # 用字典模拟数据库，key为股票代码，value为DataFrame数据

    def save_stock_data(self, df, code):
        """
        保存股票数据到缓存
        参数：
            df: DataFrame - 待保存的股票数据
            code: str - 股票代码
        返回：
            bool - 保存成功返回True，失败返回False
        """
        try:
            # 数据校验：确保包含交易日期列
            if 'trade_date' not in df.columns:
                return False
            self.data_cache[code] = df.copy()  # 深拷贝避免原数据被修改
            return True
        except Exception as e:
            print(f"保存数据失败: {e}")  # 控制台输出错误信息
            return False

    def load_stock_data(self, code):
        """
        从缓存加载股票数据
        参数：
            code: str - 股票代码
        返回：
            DataFrame/None - 成功返回数据，失败/无数据返回None
        """
        return self.data_cache.get(code, None)  # 从字典中获取数据，无则返回None


class DataFetcher:
    """
    数据获取类（模拟Tushare接口）
    功能：模拟爬取股票数据，生成符合格式的测试数据
    """

    def get_stock_data(self, code, start_date, end_date):
        """
        模拟获取股票日线数据
        参数：
            code: str - 股票代码
            start_date: str - 开始日期（格式：YYYY-MM-DD）
            end_date: str - 结束日期（格式：YYYY-MM-DD）
        返回：
            DataFrame/None - 生成的模拟数据，失败返回None
        """
        try:
            # 生成交易日历（仅工作日）
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            n_days = len(dates)  # 获取总交易日数

            # 生成模拟股价数据：基于随机游走模型
            base_price = 100 + np.random.randint(-20, 20)  # 基础价格
            changes = np.random.randn(n_days) * 2  # 日价格变动
            prices = base_price + np.cumsum(changes)  # 累计得到收盘价序列

            # 生成其他模拟数据
            volumes = np.random.randint(1000000, 10000000, size=n_days)  # 成交量
            pct_chg = (prices[1:] / prices[:-1] - 1) * 100  # 涨跌幅
            pct_chg = np.append(pct_chg, pct_chg[-1])  # 补全最后一天数据

            # 创建结构化DataFrame，模拟真实股票数据格式
            df = pd.DataFrame({
                'trade_date': dates.strftime('%Y%m%d'),  # 交易日期格式化
                'open': prices * (1 + np.random.randn(n_days) * 0.01),  # 开盘价（小幅随机）
                'high': prices * (1 + np.random.randn(n_days) * 0.02 + 0.01),  # 最高价
                'low': prices * (1 - np.random.randn(n_days) * 0.02 - 0.01),  # 最低价
                'close': prices,  # 收盘价
                'vol': volumes,  # 成交量
                'pct_chg': pct_chg  # 涨跌幅
            })

            return df.round(2)  # 保留两位小数，模拟真实数据精度
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def get_stock_basic(self, code):
        """
        获取股票基本信息（模拟）
        参数：
            code: str - 股票代码
        返回：
            DataFrame/None - 股票基本信息
        """
        try:
            # 构造模拟的股票基本信息
            data = {
                'name': f'模拟股票{code}',
                'list_date': '20100101',
                'market': '沪深A股',
                'industry': '信息技术'
            }
            return pd.DataFrame([data])  # 转为DataFrame格式
        except:
            return None


class Experiment2:
    """
    实验二核心业务类
    功能：整合数据爬取、数据库操作、数据清洗、机器学习预测全流程
    """

    def __init__(self, db_manager, data_fetcher):
        """
        初始化实验二模块
        参数：
            db_manager: DatabaseManager - 数据库管理器实例
            data_fetcher: DataFetcher - 数据获取器实例
        """

        # 初始化核心组件
        self.db_manager = db_manager  # 数据库管理实例
        self.data_fetcher = data_fetcher  # 数据获取实例

        self.predictor = CompleteStockAnalyzer()  # 机器学习分析器实例
        self.visualizer = MLVisualizer()  # 可视化工具实例

        # 状态变量
        self.current_data = None  # 当前处理的股票数据
        self.current_stock_code = None  # 当前股票代码

        self.regression_results = None
        self.classification_results = None
        self.clustering_results = None
        self.clustering_data = None
        self.selected_task = '全部'

    def fetch_financial_data(self, code_entry, start_entry, end_entry, log_callback, status_callback):
        """
        获取金融数据（爬虫功能）
        参数：
            code_entry: Entry - 股票代码输入框
            start_entry: Entry - 开始日期输入框
            end_entry: Entry - 结束日期输入框
            log_callback: function - 日志回调函数
            status_callback: function - 状态回调函数
        """

        def fetch_thread():
            """子线程执行数据获取，避免UI卡顿"""
            status_callback('正在获取数据...')  # 更新UI状态
            # 获取输入框内容并去除首尾空格
            code = code_entry.get().strip()
            start = start_entry.get().strip()
            end = end_entry.get().strip()

            # 输入校验
            if not code or not start or not end:
                messagebox.showerror('输入错误', '请填写完整的股票代码和日期')
                status_callback('就绪')
                return

            # 记录日志
            log_callback(f"正在获取股票 {code} 的数据...")
            log_callback(f"时间范围: {start} 至 {end}")

            # 调用数据获取接口
            df = self.data_fetcher.get_stock_data(code, start, end)
            if df is not None:
                # 保存当前数据和股票代码
                self.current_data = df
                self.current_stock_code = code

                # 记录数据基本信息
                log_callback(f"数据获取成功，共 {len(df)} 条记录")
                log_callback(f"数据期间: {df['trade_date'].iloc[0]} 至 {df['trade_date'].iloc[-1]}")

                # 生成数据清洗报告
                self.data_cleaning_report(df, log_callback)

                # 获取并显示股票基本信息
                basic_info = self.data_fetcher.get_stock_basic(code)
                if basic_info is not None:
                    log_callback(f"\n股票基本信息:")
                    for _, row in basic_info.iterrows():
                        log_callback(f"  股票名称: {row['name']}")
                        log_callback(f"  上市日期: {row['list_date']}")
                        log_callback(f"  市场类型: {row['market']}")
                        log_callback(f"  行业分类: {row['industry']}")

                status_callback('数据获取完成')  # 更新完成状态
            else:
                log_callback("数据获取失败，请检查股票代码和日期格式")
                status_callback('数据获取失败')  # 更新失败状态

        # 启动子线程执行，避免主线程阻塞
        threading.Thread(target=fetch_thread).start()

    def data_cleaning_report(self, df, log_callback):
        """
        生成数据清洗报告
        参数：
            df: DataFrame - 待清洗的股票数据
            log_callback: function - 日志回调函数
        """
        log_callback("\n=== 数据清洗报告 ===")  # 日志标题

        # 检查缺失值
        missing_values = df.isnull().sum()  # 统计各列缺失值数量
        total_missing = missing_values.sum()  # 计算总缺失值

        if total_missing > 0:
            log_callback(f"发现缺失值: {total_missing} 个")
            # 逐列显示缺失值情况
            for col, missing in missing_values.items():
                if missing > 0:
                    log_callback(f"  {col}: {missing} 个缺失值")
            # 处理缺失值：前向填充（用前一天数据填充）
            df.fillna(method='ffill', inplace=True)
            log_callback("已使用前向填充法处理缺失值")
        else:
            log_callback("数据完整，无缺失值")

        # 输出数据统计信息
        log_callback(f"\n数据统计:")
        log_callback(f"  收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        log_callback(f"  平均收盘价: {df['close'].mean():.2f}")
        log_callback(f"  平均成交量: {df['vol'].mean():.2f}")
        log_callback(f"  涨跌幅范围: {df['pct_chg'].min():.2f}% - {df['pct_chg'].max():.2f}%")

    def save_to_database(self, log_callback, status_callback):
        """
        保存数据到数据库
        参数：
            log_callback: function - 日志回调函数
            status_callback: function - 状态回调函数
        """
        # 数据校验：确保已有数据
        if self.current_data is None:
            messagebox.showwarning('警告', '请先获取数据')
            return

        def save_thread():
            """子线程执行保存操作"""
            status_callback('正在保存到数据库...')
            log_callback("正在保存数据到数据库...")

            # 调用数据库保存方法
            success = self.db_manager.save_stock_data(self.current_data, self.current_stock_code)
            if success:
                # 记录保存成功信息
                log_callback("数据成功保存到数据库")
                log_callback("表名: stock_daily")
                log_callback(f"股票代码: {self.current_stock_code}")
                log_callback(f"记录数: {len(self.current_data)}")
                status_callback('数据保存完成')
            else:
                log_callback("数据保存失败，请检查数据库连接")
                status_callback('数据保存失败')

        # 启动子线程执行
        threading.Thread(target=save_thread).start()

    def load_from_database(self, code_entry, log_callback, status_callback):
        """
        从数据库读取数据
        参数：
            code_entry: Entry - 股票代码输入框
            log_callback: function - 日志回调函数
            status_callback: function - 状态回调函数
        """

        def load_thread():
            """子线程执行加载操作"""
            status_callback('正在从数据库读取数据...')
            code = code_entry.get().strip()  # 获取股票代码

            # 输入校验
            if not code:
                messagebox.showerror('输入错误', '请输入股票代码')
                status_callback('就绪')
                return

            log_callback(f"正在从数据库读取股票 {code} 的数据...")

            # 从数据库加载数据
            df = self.db_manager.load_stock_data(code)
            if df is not None and len(df) > 0:
                # 保存当前数据和股票代码
                self.current_data = df
                self.current_stock_code = code

                # 记录加载成功信息
                log_callback(f"数据读取成功，共 {len(df)} 条记录")
                log_callback(f"数据期间: {df['trade_date'].iloc[0]} 至 {df['trade_date'].iloc[-1]}")
                log_callback("\n数据预览:")
                log_callback(df.head().to_string())  # 显示前5行数据

                status_callback('数据读取完成')
            else:
                log_callback("未找到该股票的数据，请先获取并保存数据")
                status_callback('数据读取失败')

        # 启动子线程执行
        threading.Thread(target=load_thread).start()

    def ml_prediction(self, log_callback, status_callback, plot_callback, task=None):
        """
        机器学习预测（包含三类任务：回归、分类、聚类）
        参数：
            log_callback: function - 日志回调函数
            status_callback: function - 状态回调函数
            plot_callback: function - 绘图回调函数（现在不使用）
            task: str - 指定任务类型（'回归'/'分类'/'聚类'/'全部'），默认全部
        """
        # 数据校验
        if self.current_data is None:
            messagebox.showwarning('警告', '请先获取或读取数据')
            return
        selected = (task or '全部')  # 默认执行全部任务

        def predict_thread():
            """子线程执行机器学习预测，避免UI卡顿"""
            status_callback('正在进行机器学习预测...')
            log_callback("开始机器学习预测分析...")
            log_callback("执行三类任务: 回归(股价预测)、分类(涨跌预测)、聚类(模式识别)")

            # 数据预处理：复制数据避免修改原数据
            df = self.current_data.copy()

            # 统一列名，兼容数据库读出的字段格式
            # 仅在缺少标准列名时进行重命名，避免重复处理
            if 'close' not in df.columns:
                col_map = {
                    'open_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'close_price': 'close',
                    'pre_close': 'pre_close',
                    'price_change': 'change',
                    'pct_change': 'pct_chg',
                    'volume': 'vol',
                    'amount': 'amount'
                }
                df = df.rename(columns=col_map)  # 列名映射

                # 再次校验关键列
                if 'close' not in df.columns:
                    log_callback("数据列缺失：未找到 'close' 列，请先通过'获取数据'或确保数据库表字段与算法要求一致")
                    status_callback('预测失败')
                    return

            # 日期格式处理：转换为datetime并设置为索引
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()  # 按日期排序

            # 1. 执行回归任务（股价预测）
            reg_results, reg_err = self.predictor.run_regression(df)
            if reg_results:
                log_callback("\n=== 回归任务结果 ===")
                # 输出各回归模型评估指标
                for name, metrics in reg_results.items():
                    log_callback(f"{name}: RMSE={metrics['RMSE']}, R²={metrics['R2_score']}")
                # 保存回归结果，但不立即显示
                self.regression_results = reg_results
            else:
                log_callback(f"回归任务错误: {reg_err}")
                self.regression_results = None

            # 2. 执行分类任务（涨跌预测）
            cls_results, cls_err = self.predictor.run_classification(df)
            if cls_results:
                log_callback("\n=== 分类任务结果 ===")
                # 输出各分类模型评估指标
                for name, metrics in cls_results.items():
                    log_callback(f"{name}: 准确率={metrics['accuracy']}, F1={metrics['f1_score']}")
                # 保存分类结果，但不立即显示
                self.classification_results = cls_results
            else:
                log_callback(f"分类任务错误: {cls_err}")
                self.classification_results = None

            # 3. 执行聚类任务（行情模式识别）
            clu_data = self.predictor.prepare_clustering_data(df)  # 准备聚类数据
            clu_results, clu_err = self.predictor.run_clustering(df)
            if clu_results and len(clu_data) > 0:
                log_callback("\n=== 聚类任务结果 ===")
                # 输出各聚类模型评估指标
                for name, metrics in clu_results.items():
                    log_callback(f"{name}: 聚类数={metrics['n_clusters']}, 轮廓系数={metrics['silhouette_score']}")
                # 保存聚类结果，但不立即显示
                self.clustering_results = clu_results
                self.clustering_data = clu_data
            else:
                log_callback(f"聚类任务错误: {clu_err}")
                self.clustering_results = None
                self.clustering_data = None

            # 保存选择的任务类型
            self.selected_task = selected
            
            status_callback('预测完成')  # 更新完成状态

        # 启动子线程执行
        threading.Thread(target=predict_thread).start()

    def visualize_ml_results(self):
        """
        可视化机器学习预测结果（直接显示图表）
        返回：bool - 是否成功显示
        """
        try:
            # 检查是否有预测结果
            if (self.regression_results is None and 
                self.classification_results is None and 
                self.clustering_results is None):
                messagebox.showwarning('警告', '请先执行机器学习预测')
                return False
            
            # 直接调用可视化方法
            # 1. 显示回归结果
            if self.selected_task in ('全部', '回归') and self.regression_results:
                self.visualizer.plot_regression_results(self.regression_results)
            
            # 2. 显示分类结果
            if self.selected_task in ('全部', '分类') and self.classification_results:
                self.visualizer.plot_classification_results(self.classification_results)
            
            # 3. 显示聚类结果
            if self.selected_task in ('全部', '聚类') and self.clustering_results and self.clustering_data is not None:
                self.visualizer.plot_clustering_results(self.clustering_results, self.clustering_data)
            
            return True
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return False

    def clear_data(self, log_callback):
        """
        重置当前数据
        参数：
            log_callback: function - 日志回调函数
        """
        self.current_data = None  # 清空当前数据
        self.current_stock_code = None  # 清空当前股票代码
        log_callback("数据已重置")  # 记录日志


class MainWindow:
    """
    主窗口界面类
    功能：创建完整的GUI界面，整合所有功能模块
    """

    def __init__(self, root):
        """
        初始化主窗口
        参数：
            root: Tk - TKinter根窗口对象
        """
        self.root = root
        self.root.title("股票数据分析实验平台")  # 设置窗口标题
        self.root.geometry("1200x800")  # 设置窗口大小

        # 初始化核心组件
        self.db_manager = DatabaseManager()  # 数据库管理器
        self.data_fetcher = DataFetcher()  # 数据获取器
        self.experiment = Experiment2(self.db_manager, self.data_fetcher)  # 实验二核心实例

        # 创建GUI组件
        self.create_widgets()

    def create_widgets(self):
        """创建所有GUI组件，布局管理"""
        # 主框架：包含所有组件
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板：输入和操作按钮
        control_frame = ttk.LabelFrame(main_frame, text="控制区", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 股票代码输入框
        ttk.Label(control_frame, text="股票代码:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.code_entry = ttk.Entry(control_frame, width=15)
        self.code_entry.grid(row=0, column=1, pady=5)
        self.code_entry.insert(0, "600000")  # 默认值

        # 开始日期输入框
        ttk.Label(control_frame, text="开始日期:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.start_entry = ttk.Entry(control_frame, width=15)
        self.start_entry.grid(row=1, column=1, pady=5)
        self.start_entry.insert(0, "2020-01-01")  # 默认值

        # 结束日期输入框
        ttk.Label(control_frame, text="结束日期:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.end_entry = ttk.Entry(control_frame, width=15)
        self.end_entry.grid(row=2, column=1, pady=5)
        self.end_entry.insert(0, "2023-12-31")  # 默认值

        # 按钮区域：包含所有功能按钮
        button_frame = ttk.Frame(control_frame, padding="10")
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        # 获取数据按钮
        self.fetch_btn = ttk.Button(
            button_frame, text="获取数据",
            command=lambda: self.experiment.fetch_financial_data(
                self.code_entry, self.start_entry, self.end_entry,
                self.log_callback, self.status_callback
            )
        )
        self.fetch_btn.pack(fill=tk.X, pady=2)

        # 保存到数据库按钮
        self.save_btn = ttk.Button(
            button_frame, text="保存到数据库",
            command=lambda: self.experiment.save_to_database(
                self.log_callback, self.status_callback
            )
        )
        self.save_btn.pack(fill=tk.X, pady=2)

        # 从数据库加载按钮
        self.load_btn = ttk.Button(
            button_frame, text="从数据库加载",
            command=lambda: self.experiment.load_from_database(
                self.code_entry, self.log_callback, self.status_callback
            )
        )
        self.load_btn.pack(fill=tk.X, pady=2)

        # 机器学习预测按钮
        self.predict_btn = ttk.Button(
            button_frame, text="机器学习预测",
            command=lambda: self.experiment.ml_prediction(
                self.log_callback, self.status_callback, self.plot_callback
            )
        )
        self.predict_btn.pack(fill=tk.X, pady=2)

        # 清除数据按钮
        self.clear_btn = ttk.Button(
            button_frame, text="清除数据",
            command=lambda: self.experiment.clear_data(self.log_callback)
        )
        self.clear_btn.pack(fill=tk.X, pady=2)

        # 状态显示标签
        ttk.Label(control_frame, text="状态:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.status_var = tk.StringVar(value="就绪")  # 初始状态
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=4, column=1, sticky=tk.W, pady=5)

        # 右侧显示区：日志和可视化结果
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 日志显示区域
        log_frame = ttk.LabelFrame(right_frame, text="日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, width=60, height=15)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 日志滚动条
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # 可视化结果显示区域
        self.plot_frame = ttk.LabelFrame(right_frame, text="可视化结果", padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # 初始提示文本
        self.plot_label = ttk.Label(self.plot_frame, text="请先执行机器学习预测以显示可视化结果")
        self.plot_label.pack(fill=tk.BOTH, expand=True)

    def log_callback(self, message):
        """
        日志回调函数：线程安全地更新日志显示
        参数：
            message: str - 日志信息
        """
        # 使用after方法确保在主线程更新UI
        self.root.after(0, lambda: self._update_log(message))

    def _update_log(self, message):
        """
        内部方法：更新日志文本框
        参数：
            message: str - 日志信息
        """
        self.log_text.insert(tk.END, message + "\n")  # 添加日志信息
        self.log_text.see(tk.END)  # 自动滚动到最后一行

    def status_callback(self, status):
        """
        状态回调函数：线程安全地更新状态显示
        参数：
            status: str - 状态信息
        """
        self.root.after(0, lambda: self.status_var.set(status))

    def plot_callback(self, img, title):
        """
        图像显示回调函数：线程安全地更新可视化结果
        参数：
            img: Image - PIL图像对象
            title: str - 图表标题
        """
        self.root.after(0, lambda: self._update_plot(img, title))

    def _update_plot(self, img, title):
        """
        内部方法：更新可视化结果显示
        参数：
            img: Image - PIL图像对象
            title: str - 图表标题
        """
        # 清空现有内容
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # 显示标题
        ttk.Label(self.plot_frame, text=title, font=("SimHei", 12, "bold")).pack(pady=5)

        # 显示图像
        if img:
            # 获取容器尺寸，计算缩放比例
            max_width = self.plot_frame.winfo_width() - 40
            max_height = self.plot_frame.winfo_height() - 40

            # 计算等比例缩放因子（不放大图像）
            width_ratio = max_width / img.width
            height_ratio = max_height / img.height
            ratio = min(width_ratio, height_ratio, 1.0)

            # 调整图像大小
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)  # 高质量缩放

            # 转换为TKinter图像对象
            tk_img = ImageTk.PhotoImage(image=img)
            lbl = ttk.Label(self.plot_frame, image=tk_img)
            lbl.image = tk_img  # 保持引用防止被垃圾回收
            lbl.pack(fill=tk.BOTH, expand=True)
        else:
            # 图像生成失败提示
            ttk.Label(self.plot_frame, text="无法生成可视化图像").pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    """程序入口：创建并运行主窗口"""
    root = tk.Tk()  # 创建TKinter根窗口
    app = MainWindow(root)  # 初始化主窗口应用
    root.mainloop()  # 启动主事件循环

