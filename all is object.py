# python中一切皆对象，函数和类也是对象，是python的一等公民
def ask(name="lorry"):
    print(name)

class Person:
    def __init__(self):
        print("lorry1")

# 1.赋值给一个变量
my_func = ask
my_func("lorry")
my_class = Person
my_class()
# 2. 添加到集合对象中
obj_list = []
obj_list.append(ask)
obj_list.append(Person)
for item in obj_list:
    print(item())
# 3.作为参数传递给函数
def print_type(item):
    print(type(item))
# 4.当作函数的返回值
def decorator_func():
    print("dec start")
    return ask
my_ask = decorator_func()
my_ask("TOM")
