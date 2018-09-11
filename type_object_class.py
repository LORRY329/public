# type object和class之间的关系
a = 1
b = "abc"
print(type(1))
print(type(int))
print(type("abc"))
print(type(str))
#type->int->1
#type->str->obj

class Student:
    pass

class MyStudent(Student):
    pass

stu = Student()
print(type(stu))
print(type(Student))
#type->class->obj
print(Student.__bases__)
#object是最顶层基类obj->class
#type是一个类也是一个对象
print(type.__bases__)
print(type(object))
print(object.__bases__)
#type<->object
#一切皆对象，一切都是type的实例都是type的对象
#object是所有类的基类，type也继承自object