
try:
    import pymysql  # type: ignore
except ImportError:
    pymysql = None

import pandas as pd
from tkinter.messagebox import showinfo

class DatabaseManager:
    def __init__(self):
        self.host = 'localhost'
        self.user = 'root'
        self.password = 'root123'
        self.database = 'finance_db'
        self.last_reg_img = None
        self.last_cls_img = None
        self.last_clu_img = None
    
    def connect(self):
        """连接数据库"""
        # 未安装 pymysql 时给出友好提示并返回 None
        if pymysql is None:
            showinfo('数据库错误', '未安装 pymysql，请先安装: pip install pymysql')
            return None
        try:
            conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4'
            )
            return conn
        except Exception as e:
            showinfo('数据库错误', f'连接失败: {str(e)}')
            return None
    
    def create_tables(self):
        """创建数据表"""
        conn = self.connect()
        if conn is None:
            return False
        
        try:
            cur = conn.cursor()
            
            # 创建股票日线数据表
            stock_table_sql = """
            CREATE TABLE IF NOT EXISTS stock_daily (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ts_code VARCHAR(20),
                trade_date VARCHAR(10),
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                pre_close FLOAT,
                price_change FLOAT,
                pct_change FLOAT,
                volume FLOAT,
                amount FLOAT,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cur.execute(stock_table_sql)
            
            # 创建预测结果表
            predict_table_sql = """
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ts_code VARCHAR(20),
                algorithm VARCHAR(50),
                mse_score FLOAT,
                prediction_date VARCHAR(10),
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cur.execute(predict_table_sql)
            
            conn.commit()
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            showinfo('建表错误', f'创建表失败: {str(e)}')
            return False
    
    def save_stock_data(self, df, ts_code):
        """保存股票数据到数据库"""
        conn = self.connect()
        if conn is None:
            return False
        
        try:
            cur = conn.cursor()
            
            # 清空该股票的历史数据
            cur.execute("DELETE FROM stock_daily WHERE ts_code = %s", (ts_code,))
            
            # 插入新数据
            for _, row in df.iterrows():
                sql = """
                INSERT INTO stock_daily 
                (ts_code, trade_date, open_price, high_price, low_price, close_price, 
                 pre_close, price_change, pct_change, volume, amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(sql, (
                    ts_code, str(row['trade_date']), row['open'], row['high'], 
                    row['low'], row['close'], row['pre_close'], row['change'], 
                    row['pct_chg'], row['vol'], row['amount']
                ))
            
            conn.commit()
            cur.close()
            conn.close()
            showinfo('存储成功', f'股票{ts_code}数据已保存到数据库')
            return True
            
        except Exception as e:
            showinfo('存储错误', f'保存数据失败: {str(e)}')
            return False
    
    def load_stock_data(self, ts_code):
        """从数据库加载股票数据"""
        conn = self.connect()
        if conn is None:
            return None
        
        try:
            sql = "SELECT * FROM stock_daily WHERE ts_code = %s ORDER BY trade_date"
            df = pd.read_sql(sql, conn, params=(ts_code,))
            conn.close()
            return df
        except Exception as e:
            showinfo('查询错误', f'加载数据失败: {str(e)}')
            return None