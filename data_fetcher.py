import tushare as ts
import pandas as pd
from tkinter.messagebox import showinfo


class DataFetcher:
    """金融数据获取类：集成Tushare股票数据接口"""

    def __init__(self):
        # 初始化Tushare token（需替换为个人token）
        self.token = '8e47efe12e4df9f247f61ef2ca1304932202cd7b0fc9de01c970d73f'
        ts.set_token(self.token)
        self.pro = ts.pro_api()

    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票日线数据"""
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                showinfo('数据错误', '未找到股票数据，请检查代码和日期')
                return None

            # 转换日期格式为标准格式
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
            return df.sort_values('trade_date')

        except Exception as e:
            showinfo('获取错误', f'获取股票数据失败: {str(e)}')
            return None

    def get_stock_basic(self, ts_code):
        """获取股票基本信息"""
        try:
            df = self.pro.stock_basic(ts_code=ts_code)
            return df
        except Exception as e:
            showinfo('基本信息错误', f'获取股票基本信息失败: {str(e)}')
            return None