# 数据封装和私有属性
from chapter03.class_method import Date
# __birthday是类的私有属性只有类中的公共函数可以访问
class User:
    def __init__(self, birthday):
        self.__birthday = birthday

    def get_age(self):
        return 2018 - self.__birthday.year

if __name__ == "__main__":
    user = User(Date(1995,3,29))
    # print(user.birthday)
    # 可以通过_classname__attr来访问私有属性
    print(user._User__birthday)
    print(user.get_age())