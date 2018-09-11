# 鸭子类型和多态
# 鸭子类型：当看到一只鸟走起来像鸭子，游泳起来像鸭子，叫起来也像鸭子，那么这只鸟就可以被称为鸭子
# 要实现多态要在不同类之间定义同样的方法say()
class Cat(object):
    def say(self):
        print("I am a cat")

class Dog(object):
    def say(self):
        print("I am a dog")

class Duck(object):
    def say(self):
        print("I am a duck")

animal_list = [Cat, Dog, Duck]
for animal in animal_list:
    animal().say()

a = ["lorry1", "lorry2"]
b = ["lorry2", "lorry1"]
name_tuple = ["lorry3", "lorry4"]
name_set = set()
name_set.add("lorry5")
name_set.add("lorry6")
# 可接受的参数是可迭代的对象
# 魔法函数充分利用了鸭子类型，在任意类中均可定义魔法函数
a.extend(name_tuple)
print(a)