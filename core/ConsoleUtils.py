import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QProgressBar)


class ConsoleHandler(QObject, logging.Handler):
    append_log = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.append_log.connect(parent.log_to_console)

    def emit(self, record):
        msg = self.format(record)
        self.append_log.emit(msg)


class CommandProcessor(QObject):
    command_executed = pyqtSignal(str, list)  # command, args
    terminate_requested = pyqtSignal()
    save_config_requested = pyqtSignal()
    load_config_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.commands = {
            'help': self.show_help,
            'set_log_path': self.set_log_path,
            'show_log_path': self.show_log_path,
            'clear': self.clear_console,
            'save_config': self.save_config,
            'load_config': self.load_config,
            'stop': self.stop_calculation
        }

    def process_command(self, command):
        """处理输入的命令"""
        logging.info(f"执行命令: {command}")

        cmd_parts = command.strip().split()
        if not cmd_parts:
            return

        cmd = cmd_parts[0].lower()
        args = cmd_parts[1:]

        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            logging.warning(f"未知命令: {command}。输入'help'查看可用命令。")

    def show_help(self, args=None):
        """显示帮助信息"""
        help_text = """
        可用命令:
        help - 显示此帮助信息
        set_log_path <路径> - 更改日志文件保存路径
        show_log_path - 显示当前日志文件路径
        clear - 清除控制台输出
        save_config - 保存当前配置
        load_config <预设名> - 加载预设参数
        stop - 终止当前计算(ESC键也可终止)
        """
        logging.info(help_text.strip())

    def set_log_path(self, args):
        """更改日志文件路径"""
        if not args:
            logging.error("需要提供新的日志路径")
            return

        new_path = args[0]
        try:
            new_path = os.path.abspath(new_path)
            log_dir = os.path.dirname(new_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.parent.log_file = new_path
            self.parent.setup_logging()
            logging.info(f"日志文件路径已更改为: {new_path}")
        except Exception as e:
            logging.error(f"更改日志路径失败: {str(e)}")

    def show_log_path(self, args=None):
        """显示当前日志文件路径"""
        logging.info(f"当前日志文件路径: {self.parent.log_file}")

    def clear_console(self, args=None):
        """清除控制台输出"""
        self.parent.console_output.clear()

    def save_config(self, args=None):
        """保存当前配置"""
        self.save_config_requested.emit()
        logging.info("配置保存请求已发送")

    def load_config(self, args):
        """加载预设参数"""
        if not args:
            logging.error("需要提供预设名称")
            return
        preset_name = args[0]
        self.load_config_requested.emit(preset_name)
        logging.info(f"加载预设参数请求已发送: {preset_name}")

    def stop_calculation(self, args=None):
        """终止当前计算"""
        self.terminate_requested.emit()
        logging.info("计算终止请求已发送")


class ConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 日志输出区域
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: black;
                color: #00FF00;
                font-family: Consolas;
                font-size: 10pt;
            }
        """)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 10px;
            }
        """)
        self.progress_bar.hide()

        # 命令输入区域
        command_layout = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.returnPressed.connect(self.execute_command)
        self.command_input.keyPressEvent = self.custom_key_press_event

        self.send_button = QPushButton("执行")
        self.send_button.clicked.connect(self.execute_command)

        command_layout.addWidget(self.command_input)
        command_layout.addWidget(self.send_button)

        layout.addWidget(self.console_output)
        layout.addWidget(self.progress_bar)
        layout.addLayout(command_layout)

    def custom_key_press_event(self, event):
        """自定义按键处理，ESC键终止计算"""
        if event.key() == Qt.Key_Escape:
            self.parent.command_processor.stop_calculation()
        else:
            QLineEdit.keyPressEvent(self.command_input, event)

    def execute_command(self):
        """执行输入的命令"""
        command = self.command_input.text().strip()
        if command:
            self.parent.command_processor.process_command(command)
            self.command_input.clear()

    def update_progress(self, value, maximum=None):
        """更新进度条"""
        if maximum is not None:
            self.progress_bar.setMaximum(maximum)

        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(
            f"计算进度: {value}/{self.progress_bar.maximum()} ({value / self.progress_bar.maximum() * 100:.1f}%)")

        if value == 1:
            self.progress_bar.show()
        elif value >= self.progress_bar.maximum():
            self.progress_bar.hide()