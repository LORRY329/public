#
class A:
    pass

class B(A):
    pass

b = B()
# 判断类型时使用isinstance会沿着继承链寻找
print(isinstance(b, B))
print(isinstance(b, A))
print(type(b))
# is 表示是否是同一个对象，即id是否相同，b是B的一个实例
print(type(b) is B)  # True
print(type(b) is A)  # False