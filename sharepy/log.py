import logging
def setup_logging(log_file="processing.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 1. 定义日志格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # 2. 创建文件处理器 (保存到文件)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 3. 创建控制台处理器 (打印到屏幕)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 4. 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger