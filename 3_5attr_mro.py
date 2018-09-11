# 类和实例属性的查找顺序MRO(method resolution order )方法解析顺序
class A:
    name = "A"

    def __init__(self):
        self.name = "obj"

a = A()
print(a.name)
# DFS depth first search python2之前
# BFS breath first search
# 新式类
class D:
    pass

class C(D):
    pass

class B(D):
    pass

class A(B,C):
    pass

print(A.__mro__)