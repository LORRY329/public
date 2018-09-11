# super函数
class A:
    def __init__(self):
        print("A")

class B(A):
    def __init__(self):
        print("B")
        super().__init__()

# 重写B的构造函数之后调用super的原因
from threading import Thread
class MyThread(Thread):
    def __init__(self, name, user):
        self.user = user
        super().__init__(name = name)
# super函数的执行顺序不是调用父类的构造函数而是调用mro上一级的构造函数
class A:
    def __init__(self):
        print("A")

class B(A):
    def __init__(self):
        print("B")
        super(B, self).__init__()

class C(A):
    def __init__(self):
        print("C")
        super(C, self).__init__()

class D(B,C):
    def __init__(self):
        print("D")
        super(D, self).__init__()

if __name__ == "__main__":
    d = D()
