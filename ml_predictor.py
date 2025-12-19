import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# 全局配置：设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


# ==================== 核心分析类 ====================
class CompleteStockAnalyzer:
    """
    股价分析核心类
    属性：
        random_state: 随机种子
        scaler/label_encoder: 数据预处理工具
        *_models: 各类任务模型字典
    """

    def __init__(self, random_state=42):
        """初始化分析器，加载三类任务的默认模型"""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # 1. 回归模型（股价数值预测）
        self.regression_models = {
            '线性回归': LinearRegression(n_jobs=-1),
            '随机森林回归': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
            'SVM回归': SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)
        }

        # 2. 分类模型（股价涨跌预测）
        self.classification_models = {
            '逻辑回归': LogisticRegression(
                max_iter=1000, random_state=random_state, n_jobs=-1
            ),
            '随机森林分类': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
            'SVM分类': SVC(kernel='rbf', C=10.0, gamma='scale', random_state=random_state, probability=True)
        }

        # 3. 聚类模型（股价模式识别）
        self.clustering_models = {
            'K-Means聚类': KMeans(n_clusters=3, random_state=random_state, n_init=10),
            'DBSCAN聚类': DBSCAN(eps=0.5, min_samples=5),
            '层次聚类': AgglomerativeClustering(n_clusters=3)
        }

    def calculate_technical_indicators(self, df):
        """计算技术指标（MA/RSI/波动率等），为模型提供特征"""
        df = df.copy()

        # 基础价格指标
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=5).std()

        # RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 价格变化率
        df['price_change'] = df['close'].diff()

        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    # ==================== 回归任务（股价数值预测） ====================
    def prepare_regression_data(self, df, lookback=5):
        """准备回归任务数据：构建滚动窗口特征，预测次日收盘价"""
        df = self.calculate_technical_indicators(df)

        feature_cols = ['close', 'ma5', 'ma10', 'return', 'volatility', 'rsi']
        features = []
        targets = []

        for i in range(lookback, len(df)):
            feature_window = df[feature_cols].iloc[i - lookback:i].values
            feature = feature_window.flatten()
            target = df['close'].iloc[i]  # 预测下一天收盘价

            features.append(feature)
            targets.append(target)

        X = np.array(features)
        y = np.array(targets)

        # 过滤异常值
        mask = ~(np.isinf(X).any(axis=1) | np.isnan(X).any(axis=1) | np.isnan(y))
        return X[mask], y[mask]

    def run_regression(self, df, lookback=5):
        """执行回归任务，返回各模型评估指标和预测结果"""
        try:
            X, y = self.prepare_regression_data(df, lookback)
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, test_idx = list(tscv.split(X))[-1]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            results = {}
            for name, model in self.regression_models.items():
                if name == 'SVM回归':
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # 计算回归指标
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results[name] = {
                    'MSE': round(mse, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'R2_score': round(r2, 4),
                    'predictions': y_pred,
                    'actual': y_test
                }

            return results, None
        except Exception as e:
            return None, f"回归任务失败: {str(e)}"

    # ==================== 分类任务（股价涨跌预测） ====================
    def prepare_classification_data(self, df, lookback=5):
        """准备分类任务数据：构建特征，生成涨跌标签（1=涨，0=跌/平）"""
        df = self.calculate_technical_indicators(df)

        # 创建分类标签
        df['target'] = (df['price_change'] > 0).astype(int)

        feature_cols = ['close', 'ma5', 'ma10', 'return', 'volatility', 'rsi']
        features = []
        targets = []

        for i in range(lookback, len(df)):
            feature_window = df[feature_cols].iloc[i - lookback:i].values
            feature = feature_window.flatten()
            target = df['target'].iloc[i]  # 预测下一天涨跌

            features.append(feature)
            targets.append(target)

        X = np.array(features)
        y = np.array(targets)

        # 过滤异常值
        mask = ~(np.isinf(X).any(axis=1) | np.isnan(X).any(axis=1) | np.isnan(y))
        return X[mask], y[mask]

    def run_classification(self, df, lookback=5):
        """执行分类任务，返回各模型评估指标和预测结果"""
        try:
            X, y = self.prepare_classification_data(df, lookback)
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, test_idx = list(tscv.split(X))[-1]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            results = {}
            for name, model in self.classification_models.items():
                if name == 'SVM分类':
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                # 计算分类指标
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)

                results[name] = {
                    'accuracy': round(accuracy, 4),
                    'precision': round(report['weighted avg']['precision'], 4),
                    'recall': round(report['weighted avg']['recall'], 4),
                    'f1_score': round(report['weighted avg']['f1-score'], 4),
                    'confusion_matrix': conf_matrix,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'actual': y_test
                }

            return results, None
        except Exception as e:
            return None, f"分类任务失败: {str(e)}"

    # ==================== 聚类任务（股价模式识别） ====================
    def prepare_clustering_data(self, df, lookback=5):
        """准备聚类任务数据：提取滚动窗口统计特征并标准化"""
        df = self.calculate_technical_indicators(df)

        # 使用滚动窗口的特征统计量作为聚类特征
        feature_cols = ['close', 'ma5', 'ma10', 'return', 'volatility', 'rsi']
        cluster_features = []

        for i in range(lookback, len(df)):
            window_data = df[feature_cols].iloc[i - lookback:i]

            # 提取窗口内的统计特征
            features = []
            for col in feature_cols:
                features.extend([
                    window_data[col].mean(),
                    window_data[col].std(),
                    window_data[col].max(),
                    window_data[col].min(),
                    window_data[col].pct_change().sum()
                ])

            cluster_features.append(features)

        X = np.array(cluster_features)

        # 过滤异常值并标准化
        mask = ~(np.isinf(X).any(axis=1) | np.isnan(X).any(axis=1))
        X = X[mask]
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled

    def run_clustering(self, df, lookback=5):
        """执行聚类任务，返回各模型评估指标和聚类标签"""
        try:
            X = self.prepare_clustering_data(df, lookback)

            results = {}
            for name, model in self.clustering_models.items():
                # 执行聚类
                labels = model.fit_predict(X)

                # 计算聚类指标（跳过DBSCAN的噪声点标签-1）
                valid_labels = labels[labels != -1]
                valid_X = X[labels != -1]

                if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 0:
                    silhouette = silhouette_score(valid_X, valid_labels)
                    calinski_harabasz = calinski_harabasz_score(valid_X, valid_labels)
                    cluster_counts = pd.Series(labels).value_counts().to_dict()
                else:
                    silhouette = 0
                    calinski_harabasz = 0
                    cluster_counts = pd.Series(labels).value_counts().to_dict()

                results[name] = {
                    'silhouette_score': round(silhouette, 4),
                    'calinski_harabasz_score': round(calinski_harabasz, 4),
                    'cluster_counts': cluster_counts,
                    'labels': labels,
                    'n_clusters': len(np.unique(labels)) if name != 'DBSCAN聚类' else len(np.unique(labels)) - (
                        1 if -1 in labels else 0)
                }

            return results, None
        except Exception as e:
            return None, f"聚类任务失败: {str(e)}"


# ==================== 可视化类 ====================
class MLVisualizer:
    """模型结果可视化类：提供回归/分类/聚类任务的图表绘制"""

    def __init__(self):
        self.fig_size = (16, 12)

    # ==================== 回归任务可视化 ====================
    def plot_regression_results(self, regression_results):
        """回归结果可视化：显示最佳模型的预测vs实际值"""
        if not regression_results:
            print("回归结果为空，无法可视化")
            return
        
        # 找到最佳模型（R²最高）
        best_model = None
        best_r2 = -float('inf')
        for name, metrics in regression_results.items():
            if metrics['R2_score'] > best_r2:
                best_r2 = metrics['R2_score']
                best_model = name
        
        if best_model is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'回归预测结果 - {best_model}', fontsize=16, fontweight='bold')
        
        metrics = regression_results[best_model]
        y_actual = metrics['actual']
        y_pred = metrics['predictions']
        
        # 1. 预测vs实际值折线图
        ax1.plot(y_actual, label='实际值', color='blue', alpha=0.7, linewidth=2)
        ax1.plot(y_pred, label='预测值', color='red', alpha=0.7, linewidth=2, linestyle='--')
        ax1.set_title(f'实际值 vs 预测值 (R²={best_r2:.4f})')
        ax1.set_xlabel('样本索引')
        ax1.set_ylabel('股价')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测vs实际值散点图
        ax2.scatter(y_actual, y_pred, alpha=0.6, s=30)
        # 添加对角线
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
        ax2.set_title('预测值 vs 实际值散点图')
        ax2.set_xlabel('实际值')
        ax2.set_ylabel('预测值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # ==================== 分类任务可视化 ====================
    def plot_classification_results(self, classification_results):
        """分类结果可视化：显示最佳模型的分类结果"""
        if not classification_results:
            print("分类结果为空，无法可视化")
            return
        
        # 找到最佳模型（准确率最高）
        best_model = None
        best_accuracy = -float('inf')
        for name, metrics in classification_results.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = name
        
        if best_model is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'分类预测结果 - {best_model}', fontsize=16, fontweight='bold')
        
        metrics = classification_results[best_model]
        y_actual = metrics['actual']
        y_pred = metrics['predictions']
        
        # 1. 混淆矩阵
        conf_matrix = metrics['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'混淆矩阵 (准确率={best_accuracy:.4f})')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('实际标签')
        ax1.set_xticklabels(['跌', '涨'])
        ax1.set_yticklabels(['跌', '涨'])
        
        # 2. 预测结果对比
        n_samples = min(50, len(y_actual))
        indices = range(n_samples)
        ax2.bar([i-0.2 for i in indices], y_actual[:n_samples], width=0.4, 
                label='实际', alpha=0.7, color='blue')
        ax2.bar([i+0.2 for i in indices], y_pred[:n_samples], width=0.4, 
                label='预测', alpha=0.7, color='red')
        ax2.set_title('前50个样本的预测结果对比')
        ax2.set_xlabel('样本索引')
        ax2.set_ylabel('涨跌标签 (0=跌, 1=涨)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # ==================== 聚类任务可视化 ====================
    def plot_clustering_results(self, clustering_results, clustering_data):
        """聚类结果可视化：显示K-Means聚类结果"""
        if not clustering_results or len(clustering_data) == 0:
            print("聚类结果/数据为空，无法可视化")
            return
        
        # 使用K-Means聚类结果
        model_name = 'K-Means聚类'
        if model_name not in clustering_results:
            model_name = list(clustering_results.keys())[0]
        
        metrics = clustering_results[model_name]
        labels = metrics['labels']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'聚类分析结果 - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. TSNE降维散点图
        if len(clustering_data) > 1:
            # 如果数据维度太高，使用TSNE降维
            if clustering_data.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                data_2d = tsne.fit_transform(clustering_data[:200])  # 只显示前200个样本
                labels_2d = labels[:200]
            else:
                data_2d = clustering_data[:200]
                labels_2d = labels[:200]
            
            unique_labels = np.unique(labels_2d)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                mask = labels_2d == label
                ax1.scatter(data_2d[mask, 0], data_2d[mask, 1],
                           c=[color], label=f'聚类{label}', alpha=0.7, s=50)
            
            ax1.set_title('聚类结果散点图')
            ax1.set_xlabel('特征维度1')
            ax1.set_ylabel('特征维度2')
            ax1.legend()
        
        # 2. 聚类分布饼图
        cluster_counts = metrics['cluster_counts']
        labels_list = list(cluster_counts.keys())
        sizes = list(cluster_counts.values())
        
        # 过滤掉噪声点（标签为-1）
        if -1 in labels_list:
            noise_idx = labels_list.index(-1)
            labels_list = labels_list[:noise_idx] + labels_list[noise_idx+1:]
            sizes = sizes[:noise_idx] + sizes[noise_idx+1:]
        
        if len(labels_list) > 0:
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels_list)))
            wedges, texts, autotexts = ax2.pie(sizes, labels=[f'聚类{l}' for l in labels_list], 
                                              autopct='%1.1f%%', colors=colors)
            ax2.set_title(f'聚类样本分布 (轮廓系数={metrics["silhouette_score"]:.4f})')
        else:
            ax2.text(0.5, 0.5, '聚类结果为空', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()


# ==================== 主函数（测试运行） ====================
if __name__ == "__main__":
    # 1. 创建模拟股价数据
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(len(dates)) * 2) + 100
    df = pd.DataFrame({'close': prices}, index=dates)

    # 2. 初始化分析器和可视化器
    analyzer = CompleteStockAnalyzer()
    visualizer = MLVisualizer()

    # 3. 执行各类任务
    print("=== 执行回归任务（股价数值预测）===")
    reg_results, reg_error = analyzer.run_regression(df, lookback=5)
    if reg_results:
        for name, metrics in reg_results.items():
            print(f"\n{name}:")
            print(f"  RMSE: {metrics['RMSE']}, R2: {metrics['R2_score']}")
    else:
        print(f"回归任务错误: {reg_error}")

    print("\n=== 执行分类任务（股价涨跌预测）===")
    cls_results, cls_error = analyzer.run_classification(df, lookback=5)
    if cls_results:
        for name, metrics in cls_results.items():
            print(f"\n{name}:")
            print(f"  准确率: {metrics['accuracy']}, F1分数: {metrics['f1_score']}")
    else:
        print(f"分类任务错误: {cls_error}")

    print("\n=== 执行聚类任务（股价模式识别）===")
    clu_data = analyzer.prepare_clustering_data(df, lookback=5)
    clu_results, clu_error = analyzer.run_clustering(df, lookback=5)
    if clu_results:
        for name, metrics in clu_results.items():
            print(f"\n{name}:")
            print(f"  聚类数: {metrics['n_clusters']}, 轮廓系数: {metrics['silhouette_score']}")
    else:
        print(f"聚类任务错误: {clu_error}")

    # 4. 生成可视化图表
    if reg_results:
        visualizer.plot_regression_results(reg_results)
    if cls_results:
        visualizer.plot_classification_results(cls_results)
    if clu_results and len(clu_data) > 0:
        visualizer.plot_clustering_results(clu_results, clu_data)