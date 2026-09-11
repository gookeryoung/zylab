"""优化模块异常."""


class OptimError(Exception):
    """优化模块通用异常."""


class SurrogateError(OptimError):
    """代理模型异常."""
