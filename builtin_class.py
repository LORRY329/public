# python常见的内置类型
# 对象的三个特征：身份id，类型type，值
a = 1  # a是变量，1是对象，a指向1
print(id(a))
b = {}
print(id(b))
# 1.None对象（全局只有一个）
a = None
b = None
print(id(a), id(b))
# 2.数值类型 int float complex（复数） bool
# 3.迭代类型
# 4.序列类型 list range tuple str array 二进制序列：bytes bytearray memoryview
# 5. 映射类型dict
# 6.集合类型 set frozenset
# 7.上下文管理类型with
# 8.其他 模块类型，class和实例，函数类型，方法类型（class里定义的函数），代码类型，object类型，
# type类型，elipsis类型，notimplemented类型
