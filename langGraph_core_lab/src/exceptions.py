class InterruptError(Exception):
    """节点抛出此异常以暂停图执行，等待外部干预。"""
    pass