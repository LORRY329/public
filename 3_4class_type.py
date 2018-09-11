# 类变量
class A:
    aa = 1  # aa在类内部定义叫做类变量

    def __init__(self, x, y):
        # self是类的实例
        self.x = x
        self.y = y

a = A(2,3)
A.aa = 11
a.aa = 100  # 对象点aa会新建一个属性放到实例的属性中，查找时从下往上查找到a.aa=100
print(a.x, a.y, a.aa)
print(A.aa)
# print(A.x)会报错，a是A的实例，查找时会向上查找，实例里找不到会查找类变量
# A是类查找时不会向下查找自己的实例
