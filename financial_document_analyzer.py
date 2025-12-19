import pdfplumber
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imageio.v2 import imread
import pandas as pd
import os
from tkinter import filedialog
from snownlp import SnowNLP
import numpy as np


class FinancialDocumentAnalyzer:
    """
    金融文献分析核心类
    属性：
        current_word_counts: 词频统计结果
        current_pdf_path: 当前PDF文件路径
        current_text: PDF提取文本
        sentiment_results: 情感分析结果
        excludes: 停用词集合
    """

    def __init__(self):
        # 初始化核心属性
        self.current_word_counts = None
        self.current_pdf_path = None
        self.current_text = None
        self.sentiment_results = None
        self.excludes = set()
        # 加载停用词
        self.load_stopwords()

    def load_stopwords(self):
        """加载停用词（优先本地文件，失败则使用默认集合）"""
        try:
            with open("./assets/中文停用词.txt", "r", encoding="utf-8") as fobj:
                for line in fobj:
                    line = line.strip()
                    if line:
                        self.excludes.add(line)
        except FileNotFoundError:
            # 默认停用词集合
            default_stopwords = {'的', '了', '在', '是', '我', '有', '和', '就',
                                 '不', '人', '都', '一', '一个', '上', '也', '很',
                                 '到', '说', '要', '去', '你', '会', '着', '没有',
                                 '看', '好', '自己', '这', 'cid'}
            self.excludes = default_stopwords

    def load_document(self, log_callback=None):
        """加载PDF文件并提取文本内容"""
        try:
            # 选择PDF文件
            file_path = filedialog.askopenfilename(
                title="选择金融文献PDF文件",
                filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
            )

            if not file_path:
                return False

            if log_callback:
                log_callback(f"正在加载金融文献: {os.path.basename(file_path)}")

            self.current_pdf_path = file_path

            # 提取PDF文本
            pdf = pdfplumber.open(file_path)
            text_all = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_all.append(text)
            pdf.close()

            self.current_text = "".join(text_all)

            if log_callback:
                log_callback(f"文献加载成功，文本长度: {len(self.current_text)} 字符")

            return True

        except Exception as e:
            if log_callback:
                log_callback(f"加载文献失败: {str(e)}")
            return False

    def analyze_document(self, log_callback=None):
        """执行词频统计和情感分析"""
        if self.current_text is None:
            if log_callback:
                log_callback("请先加载文献文件")
            return False

        try:
            if log_callback:
                log_callback("开始分析文献内容...")

            # 中文分词
            words = jieba.lcut(self.current_text)
            counts = {}

            # 过滤并统计词频
            for word in words:
                if len(word) == 1 or word == "cid":
                    continue
                if word in self.excludes:
                    continue
                counts[word] = counts.get(word, 0) + 1

            self.current_word_counts = counts

            # 情感分析
            if log_callback:
                log_callback("正在进行情感分析...")
            self.sentiment_results = self.analyze_sentiment(counts)

            if log_callback:
                log_callback(f"分析完成，共识别 {len(counts)} 个独特词语")
                # 显示TOP10高频词
                top10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
                log_callback("高频词TOP10:")
                for i, (word, count) in enumerate(top10):
                    log_callback(f"  {i + 1}. {word}: {count}次")

                # 情感分析结果统计
                positive_words = [w for w, s in self.sentiment_results.items() if s > 0.6]
                negative_words = [w for w, s in self.sentiment_results.items() if s < 0.4]
                log_callback(f"积极词汇: {len(positive_words)} 个")
                log_callback(f"消极词汇: {len(negative_words)} 个")

            return True

        except Exception as e:
            if log_callback:
                log_callback(f"文献分析失败: {str(e)}")
            return False

    def analyze_sentiment(self, word_counts, top_n=50):
        """对TOPn高频词进行情感分析"""
        sentiment_scores = {}
        # 取前N个高频词
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for word, count in top_words:
            try:
                s = SnowNLP(word)
                sentiment_scores[word] = s.sentiments
            except:
                sentiment_scores[word] = 0.5

        return sentiment_scores

    def visualize_analysis(self, log_callback=None):
        """生成综合可视化图表（词云+词频+情感）"""
        if self.current_word_counts is None:
            if log_callback:
                log_callback("请先分析文献")
            return False

        try:
            if log_callback:
                log_callback("生成可视化图表...")

            # 创建1行3列子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('金融文献综合分析', fontsize=16, fontweight='bold')

            # 词云图
            self.plot_wordcloud_subplot(axes[0])
            # 词频柱状图
            self.plot_word_frequency_subplot(axes[1])
            # 情感分析图
            self.plot_sentiment_analysis_subplot(axes[2])

            plt.tight_layout()
            plt.show()

            if log_callback:
                log_callback("可视化图表显示完成")

            return True

        except Exception as e:
            if log_callback:
                log_callback(f"可视化失败: {str(e)}")
            return False

    def plot_wordcloud_subplot(self, ax):
        """绘制词云子图"""
        try:
            # 加载词云掩码
            try:
                pic = imread('./assets/img/cloud.png')
            except:
                pic = None

            wc = WordCloud(
                mask=pic,
                font_path='msyh.ttc',
                repeat=False,
                background_color='white',
                max_words=100,
                max_font_size=100,
                min_font_size=10,
                random_state=50,
                scale=8
            )

            wc.generate_from_frequencies(self.current_word_counts)

            ax.imshow(wc)
            ax.set_title('词云分析', fontsize=14, fontweight='bold')
            ax.axis("off")

        except Exception as e:
            ax.text(0.5, 0.5, f'词云生成失败:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('词云分析', fontsize=14, fontweight='bold')
            ax.axis("off")

    def plot_word_frequency_subplot(self, ax):
        """绘制词频柱状子图"""
        try:
            items = list(self.current_word_counts.items())
            items.sort(key=lambda x: x[1], reverse=True)

            # 取TOP15高频词
            top15 = items[:15]
            words_top, counts_top = zip(*top15)

            bars = ax.bar(words_top, counts_top, color='skyblue', alpha=0.7)
            ax.set_title('高频词TOP15', fontsize=14, fontweight='bold')
            ax.set_xlabel("词汇")
            ax.set_ylabel("出现频次")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # 显示数值标签
            for bar, count in zip(bars, counts_top):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=9)

        except Exception as e:
            ax.text(0.5, 0.5, f'词频图生成失败:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('高频词TOP15', fontsize=14, fontweight='bold')

    def plot_sentiment_analysis_subplot(self, ax):
        """绘制情感分析子图"""
        try:
            if self.sentiment_results is None:
                ax.text(0.5, 0.5, '请先进行情感分析',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('情感分析', fontsize=14, fontweight='bold')
                return

            # 分类情感词汇
            positive_words = [(w, s) for w, s in self.sentiment_results.items() if s > 0.6]
            negative_words = [(w, s) for w, s in self.sentiment_results.items() if s < 0.4]
            neutral_words = [(w, s) for w, s in self.sentiment_results.items() if 0.4 <= s <= 0.6]

            # 绘制饼图
            categories = ['积极', '中性', '消极']
            counts = [len(positive_words), len(neutral_words), len(negative_words)]
            colors = ['#4CAF50', '#FFC107', '#F44336']

            wedges, texts, autotexts = ax.pie(
                counts,
                labels=categories,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )

            ax.set_title('前50个高频词情感分布', fontsize=14, fontweight='bold')

            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        except Exception as e:
            ax.text(0.5, 0.5, f'情感分析图生成失败:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('前50个高频词情感分布', fontsize=14, fontweight='bold')

    def generate_wordcloud(self, counts, log_callback=None):
        """单独生成词云图"""
        try:
            # 加载掩码图片
            try:
                pic = imread('./assets/img/cloud.png')
            except:
                pic = None

            wc = WordCloud(
                mask=pic,
                font_path='msyh.ttc',
                repeat=False,
                background_color='white',
                max_words=110,
                max_font_size=120,
                min_font_size=10,
                random_state=50,
                scale=10
            )

            wc.generate_from_frequencies(counts)

            plt.figure(figsize=(12, 8))
            plt.imshow(wc)
            plt.axis("off")
            plt.title("金融文献词云分析", fontsize=16)
            plt.tight_layout()
            plt.show()

            if log_callback:
                log_callback("词云图生成完成")

        except Exception as e:
            if log_callback:
                log_callback(f"词云生成失败: {str(e)}")

    def generate_word_frequency_chart(self, counts, log_callback=None):
        """单独生成词频柱状图"""
        try:
            items = list(counts.items())
            items.sort(key=lambda x: x[1], reverse=True)

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 取TOP15高频词
            top15 = items[:15]
            words_top, counts_top = zip(*top15)

            plt.figure(figsize=(12, 6))
            bars = plt.bar(words_top, counts_top, color='skyblue', alpha=0.7)
            plt.title("金融文献高频词TOP15", fontsize=16)
            plt.xlabel("词汇")
            plt.ylabel("出现频次")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # 显示数值标签
            for bar, count in zip(bars, counts_top):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plt.show()

            if log_callback:
                log_callback("词频柱状图生成完成")

        except Exception as e:
            if log_callback:
                log_callback(f"词频图生成失败: {str(e)}")

    def save_analysis_results(self, log_callback=None):
        """保存所有分析结果（CSV/PNG/TXT）"""
        if self.current_word_counts is None or not self.current_word_counts:
            if log_callback:
                log_callback("请先进行文献分析")
            return False

        try:
            if log_callback:
                log_callback("正在保存文献分析结果...")

            # 创建结果目录
            result_dir = "result/文献分析结果"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            file_name = os.path.basename(self.current_pdf_path).split('.')[0]

            # 保存词频数据
            word_freq_path = os.path.join(result_dir, f"{file_name}_词频统计.csv")
            word_items = list(self.current_word_counts.items())
            word_items.sort(key=lambda x: x[1], reverse=True)
            word_df = pd.DataFrame(word_items, columns=['词语', '频次'])
            word_df.to_csv(word_freq_path, index=False, encoding='utf-8-sig')

            # 保存情感分析结果
            if self.sentiment_results:
                sentiment_path = os.path.join(result_dir, f"{file_name}_情感分析.csv")
                sentiment_items = list(self.sentiment_results.items())
                sentiment_df = pd.DataFrame(sentiment_items, columns=['词语', '情感得分'])
                sentiment_df.to_csv(sentiment_path, index=False, encoding='utf-8-sig')

            # 保存分析报告
            report_path = os.path.join(result_dir, f"{file_name}_分析报告.txt")
            report_content = self.generate_document_report(word_items)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            # 保存各类图表
            self.save_wordcloud_image(os.path.join(result_dir, f"{file_name}_词云图.png"))
            self.save_word_frequency_chart(word_items, os.path.join(result_dir, f"{file_name}_词频图.png"))
            self.save_sentiment_chart(os.path.join(result_dir, f"{file_name}_情感分析图.png"))
            self.save_combined_chart(os.path.join(result_dir, f"{file_name}_综合分析图.png"))

            if log_callback:
                log_callback(f"文献分析结果已保存到: {result_dir}")

            return True

        except Exception as e:
            if log_callback:
                log_callback(f"保存文献分析结果失败: {str(e)}")
            return False

    def generate_document_report(self, word_items):
        """生成文本格式的分析报告"""
        total_words = sum(self.current_word_counts.values())
        unique_words = len(self.current_word_counts)

        report_content = f"金融文献分析报告\n"
        report_content += f"=" * 50 + "\n"
        report_content += f"分析文件: {os.path.basename(self.current_pdf_path)}\n"
        report_content += f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"总词频数: {total_words}\n"
        report_content += f"独特词语数: {unique_words}\n\n"

        # 情感分析统计
        if self.sentiment_results:
            positive_count = len([s for s in self.sentiment_results.values() if s > 0.6])
            negative_count = len([s for s in self.sentiment_results.values() if s < 0.4])
            neutral_count = len(self.sentiment_results) - positive_count - negative_count

            report_content += "情感分析统计:\n"
            report_content += f"积极词汇: {positive_count} 个\n"
            report_content += f"中性词汇: {neutral_count} 个\n"
            report_content += f"消极词汇: {negative_count} 个\n\n"

        # TOP20高频词
        report_content += "高频词TOP20:\n"
        report_content += "-" * 30 + "\n"
        for i, (word, count) in enumerate(word_items[:20]):
            report_content += f"{i + 1:2d}. {word:10s} : {count:4d} 次\n"

        # 词频分布统计
        report_content += f"\n词频分布统计:\n"
        report_content += f"出现1次的词语: {len([w for w, c in word_items if c == 1])} 个\n"
        report_content += f"出现2-5次的词语: {len([w for w, c in word_items if 2 <= c <= 5])} 个\n"
        report_content += f"出现6-10次的词语: {len([w for w, c in word_items if 6 <= c <= 10])} 个\n"
        report_content += f"出现10次以上的词语: {len([w for w, c in word_items if c > 10])} 个\n"

        return report_content

    def save_wordcloud_image(self, file_path):
        """保存词云图到指定路径"""
        try:
            plt.ioff()
            try:
                pic = imread('./assets/img/cloud.png')
            except:
                pic = None

            wc = WordCloud(
                mask=pic,
                font_path='msyh.ttc',
                repeat=False,
                background_color='white',
                max_words=110,
                max_font_size=120,
                min_font_size=10,
                random_state=50,
                scale=10
            )

            wc.generate_from_frequencies(self.current_word_counts)

            plt.figure(figsize=(12, 8))
            plt.imshow(wc)
            plt.axis("off")
            plt.title("金融文献词云分析", fontsize=16)
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            plt.ion()
            return True

        except Exception as e:
            print(f"保存词云图失败: {str(e)}")
            plt.ion()
            return False

    def save_word_frequency_chart(self, word_items, file_path):
        """保存词频柱状图到指定路径"""
        try:
            plt.ioff()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 取TOP15高频词
            top15 = word_items[:15]
            words_top, counts_top = zip(*top15)

            plt.figure(figsize=(12, 6))
            bars = plt.bar(words_top, counts_top, color='skyblue', alpha=0.7)
            plt.title("金融文献高频词TOP15", fontsize=16)
            plt.xlabel("词汇")
            plt.ylabel("出现频次")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # 显示数值标签
            for bar, count in zip(bars, counts_top):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            plt.ion()
            return True

        except Exception as e:
            print(f"保存词频图失败: {str(e)}")
            plt.ion()
            return False

    def save_sentiment_chart(self, file_path):
        """保存情感分析饼图到指定路径"""
        try:
            plt.ioff()
            if not self.sentiment_results:
                return False

            # 分类情感词汇
            positive_words = [(w, s) for w, s in self.sentiment_results.items() if s > 0.6]
            negative_words = [(w, s) for w, s in self.sentiment_results.items() if s < 0.4]
            neutral_words = [(w, s) for w, s in self.sentiment_results.items() if 0.4 <= s <= 0.6]

            categories = ['积极', '中性', '消极']
            counts = [len(positive_words), len(neutral_words), len(negative_words)]
            colors = ['#4CAF50', '#FFC107', '#F44336']

            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie(
                counts,
                labels=categories,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )

            plt.title('金融文献情感分布分析', fontsize=16)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            plt.ion()
            return True

        except Exception as e:
            print(f"保存情感分析图失败: {str(e)}")
            plt.ion()
            return False

    def save_combined_chart(self, file_path):
        """保存综合分析图（词云+词频+情感）到指定路径"""
        try:
            plt.ioff()
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('金融文献综合分析', fontsize=16, fontweight='bold')

            self.plot_wordcloud_subplot(axes[0])
            self.plot_word_frequency_subplot(axes[1])
            self.plot_sentiment_analysis_subplot(axes[2])

            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            plt.ion()
            return True

        except Exception as e:
            print(f"保存综合分析图失败: {str(e)}")
            plt.ion()
            return False

    def clear_document_data(self, log_callback=None):
        """重置所有文献分析数据"""
        self.current_word_counts = None
        self.current_pdf_path = None
        self.current_text = None
        self.sentiment_results = None
        if log_callback:
            log_callback("文献数据已重置")