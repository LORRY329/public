# 抽象基类(abc模块)
# python 是动态语言没有变量的类型，可以随时修改
# 应用场景1.检查某个类是否有某种方法

class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list

    def __len__(self):
        return len(self.employee)

com = Company(["lorry1", "lorry2"])
print(hasattr(com, "__len__"))
# 判定某个对象的类型
from collections.abc import Sized
isinstance(com, Sized)

# 强制某个子类必须实现某些方法
# 实现了一个web框架，集成cache(redis,cache,memorychache)
# 需要设计一个抽象基类， 制定子类必须实现某些方法
import abc
from collections.abc import *

class CacheBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get(self, key):
        pass

    @abc.abstractmethod
    def set(self, key, value):
        pass

# class CacheBase():
#     def get(self, key):
#         raise NotImplementedError
#     def set(self, key, value):
#         raise NotImplementedError
#
class RedisCache(CacheBase):
    def set(self, key, value):
        pass
#
redis_cache = RedisCache()
# redis_cache.set("key", "value")
class A:
    pass

class B(A):
    pass

b = B()
isinstance(b, A)
# 抽象基类优点1.isinstance，2.强制接口