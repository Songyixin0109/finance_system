import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from time import strftime
import threading

# 导入自定义模块
from database import DatabaseManager
from data_fetcher import DataFetcher
from visualizer import DataVisualizer
from ml_predictor import CompleteStockAnalyzer
from experiment1 import Experiment1
from experiment2 import Experiment2
analyzer = CompleteStockAnalyzer()
class FinanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("金融数据挖掘系统 v3.0 - 实验一&实验二")
        self.root.geometry("1100x750")
        
        # 初始化模块
        self.db_manager = DatabaseManager()
        self.data_fetcher = DataFetcher()
        self.visualizer = DataVisualizer()
        self.predictor = CompleteStockAnalyzer()
        
        # 初始化实验模块
        self.exp1 = Experiment1(self.visualizer)
        self.exp2 = Experiment2(self.db_manager, self.data_fetcher)
        
        self.setup_ui()
        self.update_clock()
        
        # 初始化绘图组件
        self.plot_label = None
        self.plot_title = None
        self.plot_container = None
        
        # 初始化数据库表
        self.db_manager.create_tables()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建选项卡
        tab_control = ttk.Notebook(self.root)
        
        # 实验一选项卡
        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='实验一：文献与数据分析')
        
        # 实验二选项卡
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text='实验二：爬虫与机器学习')
        
        tab_control.pack(expand=1, fill='both')
        
        # 设置实验一界面
        self.setup_experiment1_ui(tab1)
        
        # 设置实验二界面
        self.setup_experiment2_ui(tab2)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set('就绪')
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief='sunken',
                             anchor='w', font=('宋体', 10))
        status_bar.pack(side='bottom', fill='x')
    
    def setup_experiment1_ui(self, parent):
        """设置实验一界面"""
        # 标题
        title_label = tk.Label(parent, text='实验一：金融文献与数据分析', 
                              font=('宋体', 20, 'bold'), fg='blue')
        title_label.pack(pady=10)
        
        # 时钟
        self.clock_label1 = tk.Label(parent, text='', font=('宋体', 14))
        self.clock_label1.pack()
        
        # 按钮框架 - 文献分析
        doc_frame = tk.LabelFrame(parent, text='金融文献分析', font=('宋体', 12), padx=10, pady=10)
        doc_frame.pack(fill='x', padx=20, pady=10)
        
        # 文献分析按钮行 - 第一行
        doc_row1_frame = tk.Frame(doc_frame)
        doc_row1_frame.pack(pady=5)
        
        tk.Button(doc_row1_frame, text='加载文献文件', command=self.load_document,
                width=15, font=('宋体', 11), bg='lightblue').grid(row=0, column=0, padx=5)
        tk.Button(doc_row1_frame, text='分析文献内容', command=self.analyze_document,
                width=15, font=('宋体', 11), bg='lightyellow').grid(row=0, column=1, padx=5)
        
        # 文献分析按钮行 - 第二行
        doc_row2_frame = tk.Frame(doc_frame)
        doc_row2_frame.pack(pady=5)
        
        tk.Button(doc_row2_frame, text='可视化展示', command=self.visualize_document,
                width=15, font=('宋体', 11), bg='orange').grid(row=0, column=0, padx=5)
        tk.Button(doc_row2_frame, text='保存文献分析', command=self.save_document_analysis,
                width=15, font=('宋体', 11), bg='lightgreen').grid(row=0, column=1, padx=5)
        tk.Button(doc_row2_frame, text='重置', command=self.clear_document_data,
                width=15, font=('宋体', 11), bg='lightgray').grid(row=0, column=2, padx=5)
        
        info_label1 = tk.Label(doc_frame, text='流程：加载文献→分析内容→可视化→保存结果', 
                            font=('宋体', 10), fg='gray')
        info_label1.pack(pady=2)
        
        # 按钮框架 - 数据分析
        data_frame = tk.LabelFrame(parent, text='金融数据分析', font=('宋体', 12), padx=10, pady=10)
        data_frame.pack(fill='x', padx=20, pady=10)
        
        # 第一行按钮
        row1_frame = tk.Frame(data_frame)
        row1_frame.pack(pady=5)
        
        tk.Button(row1_frame, text='加载数据文件', command=self.load_data_file,
                 width=15, font=('宋体', 11), bg='lightblue').grid(row=0, column=0, padx=5)
        tk.Button(row1_frame, text='分析现有数据', command=self.analyze_existing_data,
                 width=15, font=('宋体', 11), bg='lightyellow').grid(row=0, column=1, padx=5)
        
        # 第二行按钮
        row2_frame = tk.Frame(data_frame)
        row2_frame.pack(pady=5)
        
        tk.Button(row2_frame, text='可视化展示', command=self.visualize_data_exp1,
                 width=15, font=('宋体', 11), bg='orange').grid(row=0, column=0, padx=5)
        tk.Button(row2_frame, text='保存分析结果', command=self.save_analysis_results,
                 width=15, font=('宋体', 11), bg='lightgreen').grid(row=0, column=1, padx=5)
        tk.Button(row2_frame, text='重置', command=self.clear_data_exp1,
                 width=15, font=('宋体', 11), bg='lightgray').grid(row=0, column=2, padx=5)
        
        info_label2 = tk.Label(data_frame, text='流程：加载文件→数据分析→可视化→保存结果', 
                              font=('宋体', 10), fg='gray')
        info_label2.pack(pady=2)
        
        # 结果显示区域
        result_frame1 = tk.Frame(parent)
        result_frame1.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.text_area1 = scrolledtext.ScrolledText(result_frame1, width=120, height=20,
                                                   font=('Consolas', 10))
        self.text_area1.pack(fill='both', expand=True)
    
    def setup_experiment2_ui(self, parent):
        """设置实验二界面"""
        # 标题
        title_label = tk.Label(parent, text='实验二：数据爬取与机器学习', 
                              font=('宋体', 20, 'bold'), fg='green')
        title_label.pack(pady=10)
        
        # 时钟
        self.clock_label2 = tk.Label(parent, text='', font=('宋体', 14))
        self.clock_label2.pack()
        
        # 输入框架
        input_frame = tk.Frame(parent)
        input_frame.pack(pady=10)
        
        # 股票代码
        tk.Label(input_frame, text='股票代码:', font=('宋体', 12)).grid(row=0, column=0, padx=5)
        self.code_entry = tk.Entry(input_frame, width=15, font=('宋体', 12))
        self.code_entry.grid(row=0, column=1, padx=5)
        self.code_entry.insert(0, '000001.SZ')
        
        # 开始日期
        tk.Label(input_frame, text='开始日期:', font=('宋体', 12)).grid(row=0, column=2, padx=5)
        self.start_entry = tk.Entry(input_frame, width=12, font=('宋体', 12))
        self.start_entry.grid(row=0, column=3, padx=5)
        self.start_entry.insert(0, '20240101')
        
        # 结束日期
        tk.Label(input_frame, text='结束日期:', font=('宋体', 12)).grid(row=0, column=4, padx=5)
        self.end_entry = tk.Entry(input_frame, width=12, font=('宋体', 12))
        self.end_entry.grid(row=0, column=5, padx=5)
        self.end_entry.insert(0, '20241231')
        
        # 功能按钮框架
        button_frame = tk.LabelFrame(parent, text='数据爬取与机器学习', font=('宋体', 12), padx=10, pady=10)
        button_frame.pack(fill='x', padx=20, pady=10)
        
        # 第一行按钮 - 数据获取
        row1_frame = tk.Frame(button_frame)
        row1_frame.pack(pady=5)
        
        tk.Button(row1_frame, text='获取数据(爬虫)', command=self.fetch_data, 
                 width=18, font=('宋体', 11), bg='lightcoral').grid(row=0, column=0, padx=5)
        tk.Button(row1_frame, text='保存到数据库', command=self.save_to_db,
                 width=18, font=('宋体', 11), bg='lightyellow').grid(row=0, column=1, padx=5)
        tk.Button(row1_frame, text='从数据库读取', command=self.load_from_db,
                 width=18, font=('宋体', 11), bg='lightgreen').grid(row=0, column=2, padx=5)
        
        # 第二行按钮 - 机器学习
        row2_frame = tk.Frame(button_frame)
        row2_frame.pack(pady=5)
        
        tk.Button(row2_frame, text='机器学习预测', command=self.ml_prediction,
                 width=18, font=('宋体', 11), bg='orange').grid(row=0, column=0, padx=5)
        tk.Button(row2_frame, text='可视化预测结果',
                 command=self.visualize_prediction,
                 width=18, font=('宋体', 11), bg='lightblue').grid(row=0, column=1, padx=5)

        tk.Button(row2_frame, text='重置数据', command=self.clear_data_exp2,
                 width=18, font=('宋体', 11), bg='lightgray').grid(row=0, column=2, padx=5)
        
        # 图表选择（全部/回归/分类/聚类）
        tk.Label(row2_frame, text='显示图表:', font=('宋体', 11)).grid(row=0, column=3, padx=5)
        self.chart_select_var = tk.StringVar(value='全部')
        ttk.Combobox(row2_frame, textvariable=self.chart_select_var, values=['全部','回归','分类','聚类'], state='readonly', width=8).grid(row=0, column=4, padx=5)
        
        info_label = tk.Label(button_frame, text='功能：数据爬取、数据库操作、机器学习算法预测', 
                            font=('宋体', 10), fg='gray')
        info_label.pack(pady=5)
        
        # 结果显示区域
        result_frame2 = tk.Frame(parent)
        result_frame2.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.text_area2 = scrolledtext.ScrolledText(result_frame2, width=120, height=25,
                                                   font=('Consolas', 10))
        self.text_area2.pack(fill='x', expand=False)
    
    def update_clock(self):
        """更新时钟"""
        current_time = strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(self, 'clock_label1'):
            self.clock_label1.config(text=current_time)
        if hasattr(self, 'clock_label2'):
            self.clock_label2.config(text=current_time)
        self.root.after(1000, self.update_clock)
    
    def log_message(self, message, area=1):
        """在文本区域显示消息"""
        if area == 1:
            self.text_area1.insert(tk.END, f"{message}\n")
            self.text_area1.see(tk.END)
        else:
            self.text_area2.insert(tk.END, f"{message}\n")
            self.text_area2.see(tk.END)
        self.root.update()
    
    def set_status(self, status):
        """设置状态栏"""
        self.status_var.set(status)
    
    # ==================== 实验一方法 ====================
    
    def load_document(self):
        """加载金融文献"""
        success = self.exp1.load_document(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.log_message("文献加载完成，请点击'分析文献内容'进行词频分析", 1)
            self.set_status('文献加载完成')
    
    def analyze_document(self):
        """分析金融文献"""
        success = self.exp1.analyze_document(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.log_message("文献分析完成，请点击'可视化展示'查看图表或'保存文献分析'保存文件", 1)
            self.set_status('文献分析完成')
    
    def visualize_document(self):
        """可视化文献分析结果"""
        success = self.exp1.visualize_document(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.set_status('文献可视化完成')
        else:
            self.set_status('文献可视化失败')
    
    def save_document_analysis(self):
        """保存文献分析结果"""
        success = self.exp1.save_document_analysis(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.set_status('文献分析结果保存完成')
        else:
            self.set_status('保存失败')
    
    def clear_document_data(self):
        """重置文献数据"""
        self.exp1.clear_document_data(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        self.set_status('文献数据已重置')
    
    def load_data_file(self):
        """加载数据文件"""
        success = self.exp1.load_data_file(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.log_message("数据文件加载完成，请点击'分析现有数据'进行数据分析", 1)
            self.set_status('数据文件加载完成')
    
    def analyze_existing_data(self):
        """分析现有数据 - 仅分析，不显示图表"""
        success = self.exp1.analyze_existing_data(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.log_message("数据分析完成，请点击'可视化分析'查看图表或'保存分析结果'保存文件", 1)
            self.set_status('数据分析完成')
    
    def visualize_data_exp1(self):
        """实验一数据可视化 - 仅显示图表，不保存"""
        success = self.exp1.visualize_data(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.set_status('图表显示完成')
        else:
            self.set_status('图表显示失败')
    
    def save_analysis_results(self):
        """实验一保存分析结果 - 保存数据和图表"""
        success = self.exp1.save_analysis_results(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        if success:
            self.set_status('分析结果保存完成')
        else:
            self.set_status('保存失败')
    
    def clear_data_exp1(self):
        """重置实验一数据"""
        self.exp1.clear_data(
            log_callback=lambda msg: self.log_message(msg, 1)
        )
        self.set_status('实验一数据已重置')
    
    # ==================== 实验二方法 ====================
    
    def fetch_data(self):
        """获取数据"""
        self.exp2.fetch_financial_data(
            self.code_entry, self.start_entry, self.end_entry,
            log_callback=lambda msg: self.log_message(msg, 2),
            status_callback=self.set_status
        )
    
    def save_to_db(self):
        """保存到数据库"""
        self.exp2.save_to_database(
            log_callback=lambda msg: self.log_message(msg, 2),
            status_callback=self.set_status
        )
    
    def load_from_db(self):
        """从数据库读取"""
        self.exp2.load_from_database(
            self.code_entry,
            log_callback=lambda msg: self.log_message(msg, 2),
            status_callback=self.set_status
        )
    
    def ml_prediction(self):
        """机器学习预测（仅运行预测，不显示图表）"""
        self.exp2.ml_prediction(
            log_callback=lambda msg: self.log_message(msg, 2),
            status_callback=self.set_status,
            plot_callback=lambda img, title: None  # 现在不使用回调
        )

    def visualize_prediction(self):
        """可视化预测结果（显示图表）"""
        # 读取用户选择的图表类型
        task = getattr(self, 'chart_select_var', None)
        task_value = task.get() if task else '全部'
        
        # 如果还没有执行过预测，需要先执行预测
        if (self.exp2.regression_results is None and 
            self.exp2.classification_results is None and 
            self.exp2.clustering_results is None):
            
            # 先执行预测
            def run_prediction_then_visualize():
                # 执行预测
                self.exp2.ml_prediction(
                    log_callback=lambda msg: self.log_message(msg, 2),
                    status_callback=self.set_status,
                    plot_callback=lambda img, title: None,
                    task=task_value
                )
                # 等待预测完成（简单延迟）
                self.root.after(2000, self.exp2.visualize_ml_results)
            
            # 启动预测线程
            threading.Thread(target=run_prediction_then_visualize).start()
        else:
            # 已经有预测结果，直接显示
            self.exp2.selected_task = task_value
            self.exp2.visualize_ml_results()
    
    def clear_data_exp2(self):
        """重置实验二数据"""
        self.exp2.clear_data(
            log_callback=lambda msg: self.log_message(msg, 2)
        )
        self.set_status('实验二数据已重置')

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = FinanceSystem(root)
        root.mainloop()
    except Exception as e:
        print(f"程序启动错误: {e}")
        messagebox.showerror("错误", f"程序启动失败: {e}")

if __name__ == "__main__":
    main()