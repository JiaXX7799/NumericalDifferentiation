import sys
import os
import numpy as np
import pandas as pd
import qtmodern.styles
import qtmodern.windows
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QSpinBox, QFileDialog,
                             QMessageBox, QGroupBox, QFormLayout, QComboBox,
                             QCheckBox, QDoubleSpinBox, QTextEdit, QSplitter)
from matplotlib import pyplot as plt, font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import scipy.signal as signal


class NumericalDifferentiator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numerical Differentiation")

        # 初始化数据
        self.time_data = None
        self.damage_data = None
        self.slope_data = None
        self.selected_method = "center"
        self.smoothed_slope = None
        self.peak_indices = None
        self.valley_indices = None

        # 控制面板组件
        self.control_group = QGroupBox("控制面板")
        self.file_label = QLabel("未选择文件")
        self.browse_btn = QPushButton("选择Excel文件")

        # 分析方法选择
        self.method_label = QLabel("微分方法:")
        self.method_combo = QComboBox()

        # 平滑选项
        self.smooth_label = QLabel("平滑窗口:")
        self.smooth_spin = QSpinBox()

        # 峰值检测选项
        self.peak_threshold_label = QLabel("峰值阈值:")
        self.peak_threshold_spin = QDoubleSpinBox()

        # 谷值检测选项
        self.valley_threshold_label = QLabel("谷值阈值:")
        self.valley_threshold_spin = QDoubleSpinBox()

        # 显示选项
        self.show_peaks_check = QCheckBox("显示峰值")
        self.show_valleys_check = QCheckBox("显示谷值")
        self.show_smoothed_check = QCheckBox("显示平滑曲线")
        self.show_original_check = QCheckBox("显示原始数据点")

        # 导出按钮
        self.export_btn = QPushButton("导出数据")

        # 分析结果组件
        self.result_group = QGroupBox("分析结果")
        self.stats_text = QTextEdit()

        # 峰值信息组件
        self.peaks_group = QGroupBox("峰值/谷值信息")
        self.peaks_text = QTextEdit()

        # 图表组件
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # 加载自定义字体文件
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HarmonyOS_SansSC_Regular.ttf")

        # 检查字体文件是否存在
        if os.path.exists(path):
            # 为matplotlib注册字体
            font_manager.fontManager.addfont(path)
            custom_font = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams['font.sans-serif'] = [custom_font, 'SimHei', 'Arial']

            self.custom_font_name = custom_font
        else:
            print(f"警告: 找不到字体文件 {path}")
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Helvetica']
            self.custom_font_name = 'Arial'

        plt.rcParams['axes.unicode_minus'] = False

        # 创建UI
        self.create_widgets()
        self.setup_layout()

        # 设置窗口初始大小
        self.resize(1800, 600)

    def create_widgets(self):
        self.file_label.setWordWrap(True)
        self.browse_btn.clicked.connect(self.load_file)

        # 设置微分方法选择
        self.method_combo.addItems(["中心差分法", "前向差分法", "后向差分法"])
        self.method_combo.currentIndexChanged.connect(self.update_method)

        # 设置平滑选项
        self.smooth_spin.setRange(1, 21)
        self.smooth_spin.setValue(5)
        self.smooth_spin.setSingleStep(2)
        self.smooth_spin.valueChanged.connect(self.update_smoothing)

        # 设置峰值检测选项
        self.peak_threshold_spin.setRange(0.01, 10.0)
        self.peak_threshold_spin.setValue(0.5)
        self.peak_threshold_spin.setSingleStep(0.1)
        self.peak_threshold_spin.valueChanged.connect(self.update_peak_detection)

        # 设置谷值检测选项
        self.valley_threshold_spin.setRange(0.01, 10.0)
        self.valley_threshold_spin.setValue(0.5)
        self.valley_threshold_spin.setSingleStep(0.1)
        self.valley_threshold_spin.valueChanged.connect(self.update_valley_detection)

        # 设置显示选项
        self.show_peaks_check.setChecked(True)
        self.show_peaks_check.stateChanged.connect(self.update_display)

        self.show_valleys_check.setChecked(True)
        self.show_valleys_check.stateChanged.connect(self.update_display)

        self.show_smoothed_check.setChecked(True)
        self.show_smoothed_check.stateChanged.connect(self.update_display)

        self.show_original_check.setChecked(True)
        self.show_original_check.stateChanged.connect(self.update_display)

        self.export_btn.clicked.connect(self.export_data)

        # 设置文本编辑框属性
        self.stats_text.setReadOnly(True)
        self.peaks_text.setReadOnly(True)

        # 设置最小尺寸
        self.stats_text.setMinimumHeight(200)
        self.peaks_text.setMinimumHeight(200)

    def setup_layout(self):
        # 设置控制面板布局
        control_layout = QFormLayout()
        control_layout.addRow(self.browse_btn)
        control_layout.addRow(QLabel("文件名:"), self.file_label)
        control_layout.addRow(self.method_label, self.method_combo)
        control_layout.addRow(self.smooth_label, self.smooth_spin)
        control_layout.addRow(self.peak_threshold_label, self.peak_threshold_spin)
        control_layout.addRow(self.valley_threshold_label, self.valley_threshold_spin)
        control_layout.addRow(self.show_peaks_check)
        control_layout.addRow(self.show_valleys_check)
        control_layout.addRow(self.show_smoothed_check)
        control_layout.addRow(self.show_original_check)
        control_layout.addRow(self.export_btn)
        self.control_group.setLayout(control_layout)
        self.control_group.setMinimumWidth(300)
        self.control_group.setMaximumWidth(400)

        # 设置分析结果布局
        result_layout = QVBoxLayout()
        result_layout.addWidget(QLabel("统计信息:"))
        result_layout.addWidget(self.stats_text)
        self.result_group.setLayout(result_layout)
        self.result_group.setMinimumWidth(300)
        self.result_group.setMaximumWidth(400)

        # 设置峰值信息布局
        peaks_layout = QVBoxLayout()
        peaks_layout.addWidget(self.peaks_text)
        self.peaks_group.setLayout(peaks_layout)
        self.peaks_group.setMinimumWidth(300)
        self.peaks_group.setMaximumWidth(400)

        # 创建右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.result_group)
        right_layout.addWidget(self.peaks_group)
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(400)

        # 使用分割器创建灵活的布局
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.control_group)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 700, 350])
        splitter.setHandleWidth(10)  # 使分割线更容易拖拽

        # 设置主布局
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel Files (*.xlsx *.xls)")

        if file_path:
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                if df.shape[1] < 2:
                    QMessageBox.warning(self, "数据错误", "Excel文件需要至少包含两列数据")
                    return

                # 提取时间和伤害数据
                self.time_data = df.iloc[:, 0].values
                self.damage_data = df.iloc[:, 1].values

                # 提取文件名并显示
                file_name = os.path.basename(file_path)
                self.file_label.setText(file_name)

                # 自动执行计算
                self.calculate_slope()

            except Exception as e:
                QMessageBox.critical(self, "读取错误", f"读取文件时出错: {str(e)}")

    def update_method(self, index):
        methods = ["center", "forward", "backward"]
        self.selected_method = methods[index]
        if self.time_data is not None and self.damage_data is not None:
            self.calculate_slope()

    def update_smoothing(self):
        if self.slope_data is not None:
            self.calculate_smoothed_slope()
            self.update_display()

    def update_peak_detection(self):
        if self.slope_data is not None:
            self.detect_peaks()
            self.update_display()

    def update_valley_detection(self):
        if self.slope_data is not None:
            self.detect_valleys()
            self.update_display()

    def update_display(self):
        if self.slope_data is not None:
            self.plot_results()

    def calculate_slope(self):
        if self.time_data is None or self.damage_data is None:
            return

        try:
            # 使用数值微分计算斜率
            self.slope_data = np.zeros_like(self.damage_data)
            n = len(self.time_data)

            if self.selected_method == "center":  # 中心差分法
                # 处理中间点
                for i in range(1, n - 1):
                    dt = self.time_data[i + 1] - self.time_data[i - 1]
                    dy = self.damage_data[i + 1] - self.damage_data[i - 1]
                    self.slope_data[i] = dy / dt

                # 处理端点（使用前向和后向差分）
                self.slope_data[0] = (self.damage_data[1] - self.damage_data[0]) / (
                            self.time_data[1] - self.time_data[0])
                self.slope_data[n - 1] = (self.damage_data[n - 1] - self.damage_data[n - 2]) / (
                            self.time_data[n - 1] - self.time_data[n - 2])

            elif self.selected_method == "forward":  # 前向差分法
                for i in range(0, n - 1):
                    dt = self.time_data[i + 1] - self.time_data[i]
                    dy = self.damage_data[i + 1] - self.damage_data[i]
                    self.slope_data[i] = dy / dt
                # 最后一个点使用后向差分
                self.slope_data[n - 1] = (self.damage_data[n - 1] - self.damage_data[n - 2]) / (
                            self.time_data[n - 1] - self.time_data[n - 2])

            else:  # 后向差分法
                for i in range(1, n):
                    dt = self.time_data[i] - self.time_data[i - 1]
                    dy = self.damage_data[i] - self.damage_data[i - 1]
                    self.slope_data[i] = dy / dt
                # 第一个点使用前向差分
                self.slope_data[0] = (self.damage_data[1] - self.damage_data[0]) / (
                            self.time_data[1] - self.time_data[0])

            # 计算平滑斜率和峰值
            self.calculate_smoothed_slope()
            self.detect_peaks()
            self.detect_valleys()

            # 更新统计信息
            self.update_statistics()

            # 绘制图表
            self.plot_results()

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算斜率过程中出错: {str(e)}")

    def calculate_smoothed_slope(self):
        if self.slope_data is None:
            return

        window_size = self.smooth_spin.value()
        if window_size > 1:
            # 使用移动平均平滑数据
            self.smoothed_slope = np.convolve(self.slope_data, np.ones(window_size) / window_size, mode='same')
        else:
            self.smoothed_slope = self.slope_data.copy()

    def detect_peaks(self):
        if self.slope_data is None:
            return

        threshold = self.peak_threshold_spin.value()

        # 检测峰值
        self.peak_indices, _ = signal.find_peaks(
            self.slope_data,
            height=threshold,
            distance=5  # 最小峰值距离
        )

        # 更新峰值信息
        self.update_peak_info()

    def detect_valleys(self):
        if self.slope_data is None:
            return

        threshold = self.valley_threshold_spin.value()

        # 检测谷值 - 找斜率数据的负峰值
        self.valley_indices, _ = signal.find_peaks(
            -self.slope_data,  # 将数据取反，找谷值
            height=threshold,
            distance=5  # 最小谷值距离
        )

        # 更新谷值信息
        self.update_valley_info()

    def update_statistics(self):
        if self.slope_data is None:
            return

        # 计算总时间（秒和分钟）
        total_time_sec = self.time_data[-1] - self.time_data[0]
        total_time_min = total_time_sec / 60.0

        # 计算平均DPS
        avg_dps = np.mean(self.slope_data)

        # 计算峰值和谷值数量
        peak_count = len(self.peak_indices) if self.peak_indices is not None else 0
        valley_count = len(self.valley_indices) if self.valley_indices is not None else 0

        stats_text = f"""
        <b>原始数据统计:</b><br>
        数据点数: {len(self.time_data)}<br>
        时间范围: {self.time_data[0]:.2f} - {self.time_data[-1]:.2f} 秒<br>
        总时长: {total_time_sec:.2f} 秒 ({total_time_min:.2f} 分钟)<br>
        总伤害: {self.damage_data[-1]:.6f}<br><br>

        <b>斜率数据统计:</b><br>
        平均DPS: {avg_dps:.6f}<br>
        最大DPS: {np.max(self.slope_data):.6f}<br>
        最小DPS: {np.min(self.slope_data):.6f}<br>
        DPS标准差: {np.std(self.slope_data):.6f}<br>
        DPS中位数: {np.median(self.slope_data):.6f}<br><br>

        <b>峰值/谷值统计:</b><br>
        峰值数量: {peak_count}<br>
        谷值数量: {valley_count}<br>
        峰值检测阈值: {self.peak_threshold_spin.value():.2f}<br>
        谷值检测阈值: {self.valley_threshold_spin.value():.2f}<br>
        """

        self.stats_text.setHtml(stats_text)

    def update_peak_info(self):
        if self.peak_indices is None:
            return

        peaks_text = "<b>峰值信息:</b><br>"

        if len(self.peak_indices) > 0:
            for i, idx in enumerate(self.peak_indices[:10]):  # 只显示前10个峰值
                time_sec = self.time_data[idx]
                time_min = time_sec / 60.0
                peaks_text += f"{i + 1}. 时间: {time_min:.2f}分 ({time_sec:.2f}秒), DPS: {self.slope_data[idx]:.6f}<br>"
        else:
            peaks_text += "未检测到明显的峰值<br>"

        self.peaks_text.setHtml(peaks_text)

    def update_valley_info(self):
        if self.valley_indices is None:
            return

        valleys_text = "<b>谷值信息:</b><br>"

        if len(self.valley_indices) > 0:
            for i, idx in enumerate(self.valley_indices[:10]):  # 只显示前10个谷值
                time_sec = self.time_data[idx]
                time_min = time_sec / 60.0
                valleys_text += f"{i + 1}. 时间: {time_min:.2f}分 ({time_sec:.2f}秒), DPS: {self.slope_data[idx]:.6f}<br>"
        else:
            valleys_text += "未检测到明显的谷值<br>"

        # 将谷值信息添加到峰值信息文本中
        current_text = self.peaks_text.toHtml()
        self.peaks_text.setHtml(current_text + "<br>" + valleys_text)

    def plot_results(self):
        if self.slope_data is None:
            return

        # 清除之前的图表
        self.figure.clear()
        font = {'family': self.custom_font_name, 'size': 10}

        # 创建两个子图
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        # 将时间转换为分钟
        time_minutes = self.time_data / 60.0

        # 绘制原始数据
        if self.show_original_check.isChecked():
            ax1.scatter(time_minutes, self.damage_data, color='blue', alpha=0.5, s=20, label='原始数据点')
        ax1.plot(time_minutes, self.damage_data, 'b-', alpha=0.7, label='总伤害')
        ax1.set_xlabel('时间 (分钟)', fontdict=font)
        ax1.set_ylabel('总伤害', fontdict=font)
        ax1.set_title('总伤害曲线', fontdict=font)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 设置X轴刻度为每分钟
        ax1.xaxis.set_major_locator(MultipleLocator(base=1.0))

        # 绘制斜率数据
        if self.show_original_check.isChecked():
            ax2.scatter(time_minutes, self.slope_data, color='red', alpha=0.5, s=20, label='原始DPS点')
        ax2.plot(time_minutes, self.slope_data, 'r-', alpha=0.5, label='原始DPS')

        # 绘制平滑曲线（如果启用）
        if self.show_smoothed_check.isChecked() and self.smoothed_slope is not None:
            ax2.plot(time_minutes, self.smoothed_slope, 'g-', linewidth=2, label='平滑DPS')

        # 标记峰值（如果启用）
        if self.show_peaks_check.isChecked() and self.peak_indices is not None and len(self.peak_indices) > 0:
            ax2.scatter(time_minutes[self.peak_indices], self.slope_data[self.peak_indices],
                        color='red', s=80, marker='^', label='峰值', zorder=5)

        # 标记谷值（如果启用）
        if self.show_valleys_check.isChecked() and self.valley_indices is not None and len(self.valley_indices) > 0:
            ax2.scatter(time_minutes[self.valley_indices], self.slope_data[self.valley_indices],
                        color='blue', s=80, marker='v', label='谷值', zorder=5)

        ax2.set_xlabel('时间 (分钟)', fontdict=font)
        ax2.set_ylabel('DPS', fontdict=font)
        ax2.set_title('DPS曲线', fontdict=font)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 设置X轴刻度为每分钟
        ax2.xaxis.set_major_locator(MultipleLocator(base=1.0))

        # 调整布局
        self.figure.tight_layout()

        # 刷新画布
        self.canvas.draw()

    def export_data(self):
        if self.slope_data is None:
            QMessageBox.warning(self, "导出错误", "没有数据可导出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV Files (*.csv)")

        if file_path:
            try:
                # 创建包含所有数据的DataFrame
                data = {
                    '时间(秒)': self.time_data,
                    '时间(分钟)': self.time_data / 60.0,
                    '总伤害': self.damage_data,
                    '伤害速率(DPS)': self.slope_data
                }

                if self.smoothed_slope is not None:
                    data['平滑伤害速率(DPS)'] = self.smoothed_slope

                df = pd.DataFrame(data)

                # 添加峰值标记
                peak_flags = np.zeros_like(self.time_data, dtype=bool)
                if self.peak_indices is not None:
                    peak_flags[self.peak_indices] = True
                df['是否为峰值'] = peak_flags

                # 添加谷值标记
                valley_flags = np.zeros_like(self.time_data, dtype=bool)
                if self.valley_indices is not None:
                    valley_flags[self.valley_indices] = True
                df['是否为谷值'] = valley_flags

                # 保存到CSV
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "导出成功", f"数据已成功导出到 {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "导出错误", f"导出数据时出错: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qtmodern.styles.dark(app)
    # 加载自定义字体
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HarmonyOS_SansSC_Regular.ttf")
    font_id = QFontDatabase.addApplicationFont(font_path)

    if font_id != -1:
        # 如果成功加载字体
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        app_font = QFont(font_family, 10)
    else:
        # 如果未能加载字体，使用系统字体
        print(f"警告: 无法加载字体文件 {font_path}")
        app_font = QFont('Arial', 10)
    app.setFont(app_font)
    window = NumericalDifferentiator()
    mw = qtmodern.windows.ModernWindow(window)

    mw.show()
    sys.exit(app.exec_())