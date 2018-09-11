# 自省是通过一定的机制查询对象的内部结构
from chapter03.class_method import Date


class Person:
    """
    人
    """
    name = "user"

class Student(Person):

    def __init__(self, school_name):
        self.school_name = school_name

if __name__ == "__main__":
    user = Student("CUHKSZ")

    # 通过__dict__查询属性
    print(user.__dict__)  # 实例的属性
    print(Person.__dict__)  # 类的属性
    user.__dict__["school_addr"] = "北京市"
    print(user.school_addr)
    print(user.name)
    # dir
    print(dir(user))
    a = [1,2]
    print(dir(a))


