# 魔法函数
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list

    # 定义__getitem__之后这个类就成为可迭代的，可以使用for循环迭代
    def __getitem__(self, item):
        return self.employee[item]

    #使用len()函数时会先查找__len__，没有的话去查找__getitem__
    def __len__(self):
        return len(self.employee)
    # 默认输出内存地址

    def __str__(self):
        return ",".join(self.employee)
    # 定义之后会输出字符串

    def __repr__(self):
        return ",".join(self.employee)

company = Company(["lorry", "lr", "lrds"])
print(company)
# compangy1 = company[:2]
# for em in company:
#     print(em)

# 魔法函数一览
# 非数学运算
# 字符串表示(__repr__, __str__)
# 集合序列相关(__len__, __getitem__, __setitem__,__deliitem__, __cotains__)
#迭代相关(__iter__, __next__)
# 可调用(__call__)
# with 上下文管理器(__enter__, __exit__)
# 数制转换 (__abs__, __bool__, __int__, __float__, __hash__, __index__)
# 元类相关(__new__, __init__)
# 属性相关(__getattr__, __setattr__, __getattribuate__, __setattribuate__, __dir__)
# 属性描述符(__get__, __set__, __delete__)
# 协程(__await__, __aiter__, __anext__, __aenter__, __aexit__)
# 数学运算
# 一元运算符、二元、算术、反向算术、增量赋值、位、反向位、增量赋值位

# 魔法函数不用显式调用，会自动调用
# len函数对内置数据类型求长度时不会去遍历，list等内置类型是用C语言写的，其中有一个长度的数据，len函数会直接读取这个数据
