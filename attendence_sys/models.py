from django.db import models

from django.contrib.auth.models import User


def user_directory_path(instance, filename):
    name, ext = filename.split(".")
    name = instance.firstname + instance.lastname
    filename = name + "." + ext
    return "Faculty_Images/{}".format(filename)


class Faculty(models.Model):

    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.CASCADE)
    firstname = models.CharField(max_length=200, null=True, blank=True)
    lastname = models.CharField(max_length=200, null=True, blank=True)
    phone = models.CharField(max_length=200, null=True)
    email = models.CharField(max_length=200, null=True)
    profile_pic = models.ImageField(
        upload_to=user_directory_path, null=True, blank=True
    )

    def __str__(self):
        return str(self.firstname + " " + self.lastname)


# def student_directory_path(instance, filename):
#     name, ext = filename.split(".")
#     name = (
#         instance.registration_id
#     )  # + "_" + instance.branch + "_" + instance.year + "_" + instance.section
#     filename = name + "." + ext
#     return "Student_Images/{}/{}/{}/{}".format(
#         instance.branch, instance.year, instance.section, filename
#     )


def employee_directory_path(instance, filename):
    name, ext = filename.split(".")
    name = instance.employee_id
    filename = name + "." + ext
    return "employee_images/{}".format(filename)


class Employee(models.Model):

    firstname = models.CharField(max_length=200, null=True, blank=True)
    lastname = models.CharField(max_length=200, null=True, blank=True)
    employee_id = models.CharField(max_length=200, null=True)
    rfid = models.CharField(max_length=200, null=True)
    company_name = models.CharField(max_length=200, null=True, blank=True)
    temperature = models.FloatField()
    spo2 = models.FloatField()
    entry_by = models.CharField(max_length=200, null=True, blank=True)
    profile_pic = models.ImageField(
        upload_to=employee_directory_path, null=True, blank=True
    )

    def __str__(self):
        return str(self.employee_id)


# class Student(models.Model):

#     BRANCH = (
#         ("CSE", "CSE"),
#         ("IT", "IT"),
#         ("ECE", "ECE"),
#         ("CHEM", "CHEM"),
#         ("MECH", "MECH"),
#         ("EEE", "EEE"),
#     )
#     YEAR = (
#         ("1", "1"),
#         ("2", "2"),
#         ("3", "3"),
#         ("4", "4"),
#     )
#     SECTION = (
#         ("A", "A"),
#         ("B", "B"),
#         ("C", "C"),
#     )

#     firstname = models.CharField(max_length=200, null=True, blank=True)
#     lastname = models.CharField(max_length=200, null=True, blank=True)
#     registration_id = models.CharField(max_length=200, null=True)
#     branch = models.CharField(max_length=100, null=True, choices=BRANCH)
#     year = models.CharField(max_length=100, null=True, choices=YEAR)
#     section = models.CharField(max_length=100, null=True, choices=SECTION)
#     profile_pic = models.ImageField(
#         upload_to=student_directory_path, null=True, blank=True
#     )

#     def __str__(self):
#         return str(self.registration_id)


# class Attendence(models.Model):
#     # faculty = models.ForeignKey(Faculty, null = True, on_delete= models.SET_NULL)
#     # student = models.ForeignKey(Student, null = True, on_delete= models.SET_NULL)
#     Faculty_Name = models.CharField(max_length=200, null=True, blank=True)
#     Student_ID = models.CharField(max_length=200, null=True, blank=True)
#     date = models.DateField(auto_now_add=True, null=True)
#     time = models.TimeField(auto_now_add=True, null=True)
#     branch = models.CharField(max_length=200, null=True)
#     year = models.CharField(max_length=200, null=True)
#     section = models.CharField(max_length=200, null=True)
#     period = models.CharField(max_length=200, null=True)
#     status = models.CharField(max_length=200, null=True, default="Absent")

#     def __str__(self):
#         return str(self.Student_ID + "_" + str(self.date) + "_" + str(self.period))


class Attendence(models.Model):
    entry_by = models.CharField(max_length=200, null=True, blank=True)
    employee_id = models.CharField(max_length=200, null=True, blank=True)
    employee_name = models.CharField(max_length=200, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    date = models.DateField(auto_now_add=True, null=True)
    time = models.TimeField(auto_now_add=True, null=True)
    status = models.CharField(max_length=200, null=True, default="Absent")

    def __str__(self):
        return str(self.employee_id)