from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse

from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

from .forms import *
from .models import Employee, Attendence
from .filters import AttendenceFilter

# from django.views.decorators import gzip

from .recognizer import Recognizer
from datetime import date

# @login_required(login_url = 'login')
# def home(request):
#     studentForm = CreateStudentForm()

#     if request.method == 'POST':
#         studentForm = CreateStudentForm(data = request.POST, files=request.FILES)
#         # print(request.POST)
#         stat = False
#         try:
#             student = Student.objects.get(registration_id = request.POST['registration_id'])
#             stat = True
#         except:
#             stat = False
#         if studentForm.is_valid() and (stat == False):
#             studentForm.save()
#             name = studentForm.cleaned_data.get('firstname') +" " +studentForm.cleaned_data.get('lastname')
#             messages.success(request, 'Student ' + name + ' was successfully added.')
#             return redirect('home')
#         else:
#             messages.error(request, 'Student with Registration Id '+request.POST['registration_id']+' already exists.')
#             return redirect('home')

#     context = {'studentForm':studentForm}
#     return render(request, 'attendence_sys/home.html', context)


@login_required(login_url="login")
def home(request):

    if request.method == "POST":
        employeeForm = CreateEmployeeForm(data=request.POST, files=request.FILES)
        stat = False
        try:
            employee = Employee.objects.get(
                employee_id=request.POST["employee_id"]
            )
            stat = True
        except:
            stat = False
        if employeeForm.is_valid() and (stat == False):
            employeeForm.save()
            name = (
                employeeForm.cleaned_data.get("firstname")
                + " "
                + employeeForm.cleaned_data.get("lastname")
            )
            messages.success(request, "Employee " + name + " was successfully added.")
            return redirect("home")
        else:
            messages.error(
                request,
                "Employee with employee Id "
                + request.POST["employee_id"]
                + " already exists.",
            )
            return redirect("home")

    context = {"employeeForm": CreateEmployeeForm()}
    return render(request, "attendence_sys/home.html", context)


def loginPage(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.info(request, "Username or Password is incorrect")

    context = {}
    return render(request, "attendence_sys/login.html", context)


@login_required(login_url="login")
def logoutUser(request):
    logout(request)
    return redirect("login")


@login_required(login_url="login")
def updateStudentRedirect(request):
    context = {}
    if request.method == "POST":
        try:
            employee_id = request.POST["employee_id"]
            employee = Employee.objects.get(employee_id=employee_id)
            updateEmployeeForm = CreateEmployeeForm(instance=employee)
            context = {
                "form": updateEmployeeForm,
                "prev_reg_id": employee_id,
                "employee": employee,
            }
        except:
            messages.error(request, "Employee Not Found")
            return redirect("home")
    return render(request, "attendence_sys/student_update.html", context)


@login_required(login_url="login")
def updateStudent(request):
    if request.method == "POST":
        context = {}
        try:
            employee = Employee.objects.get(employee_id=request.POST["prev_reg_id"])
            updateEmployeeForm = CreateEmployeeForm(
                data=request.POST, files=request.FILES, instance=employee
            )
            if updateEmployeeForm.is_valid():
                updateEmployeeForm.save()
                messages.success(request, "Updation Success")
                return redirect("home")
        except:
            messages.error(request, "Updation Unsucessfull")
            return redirect("home")
    return render(request, "attendence_sys/student_update.html", context)


# @login_required(login_url="login")
# def takeAttendence(request):
#     if request.method == "POST":
#         details = {
#             "branch": request.POST["branch"],
#             "year": request.POST["year"],
#             "section": request.POST["section"],
#             "period": request.POST["period"],
#             "faculty": request.user.faculty,
#         }
#         if (
#             Attendence.objects.filter(
#                 date=str(date.today()),
#                 branch=details["branch"],
#                 year=details["year"],
#                 section=details["section"],
#                 period=details["period"],
#             ).count()
#             != 0
#         ):
#             messages.error(request, "Attendence already recorded.")
#             return redirect("home")
#         else:
#             students = Student.objects.filter(
#                 branch=details["branch"],
#                 year=details["year"],
#                 section=details["section"],
#             )
#             names = Recognizer(details)
#             for student in students:
#                 if str(student.registration_id) in names:
#                     attendence = Attendence(
#                         Faculty_Name=request.user.faculty,
#                         Student_ID=str(student.registration_id),
#                         period=details["period"],
#                         branch=details["branch"],
#                         year=details["year"],
#                         section=details["section"],
#                         status="Present",
#                     )
#                     attendence.save()
#                 else:
#                     attendence = Attendence(
#                         Faculty_Name=request.user.faculty,
#                         Student_ID=str(student.registration_id),
#                         period=details["period"],
#                         branch=details["branch"],
#                         year=details["year"],
#                         section=details["section"],
#                     )
#                     attendence.save()
#             attendences = Attendence.objects.filter(
#                 date=str(date.today()),
#                 branch=details["branch"],
#                 year=details["year"],
#                 section=details["section"],
#                 period=details["period"],
#             )
#             context = {"attendences": attendences, "ta": True}
#             messages.success(request, "Attendence taking Success")
#             return render(request, "attendence_sys/attendence.html", context)
#     context = {}
#     return render(request, "attendence_sys/home.html", context)

@login_required(login_url="login")
def takeAttendence(request):
    if request.method == "POST":
        details = {
            "company": request.POST["company"],
        }

        names = Recognizer(details)

        company_employees = Employee.objects.filter(company=details["company"])
        for employee in company_employees:
            if str(employee.employee_id) in names:
                attendence = Attendence(
                    Faculty_Name=request.user.faculty,
                    Student_ID=str(employee.registration_id),
                    section=details["section"],
                    status="Present",
                )
                attendence.save()
            else:
                attendence = Attendence(
                    Faculty_Name=request.user.faculty,
                    Student_ID=str(student.registration_id),
                    period=details["period"],
                    branch=details["branch"],
                    year=details["year"],
                    section=details["section"],
                )
                attendence.save()
        attendences = Attendence.objects.filter(
            date=str(date.today()),
            branch=details["branch"],
            year=details["year"],
            section=details["section"],
            period=details["period"],
        )
        context = {"attendences": attendences, "ta": True}
        messages.success(request, "Attendence taking Success")
        return render(request, "attendence_sys/attendence.html", context)
    context = {}
    return render(request, "attendence_sys/home.html", context)

def searchAttendence(request):
    attendences = Attendence.objects.all()
    myFilter = AttendenceFilter(request.GET, queryset=attendences)
    attendences = myFilter.qs
    context = {"myFilter": myFilter, "attendences": attendences, "ta": False}
    return render(request, "attendence_sys/attendence.html", context)


def facultyProfile(request):
    faculty = request.user.faculty
    form = FacultyForm(instance=faculty)
    context = {"form": form}
    return render(request, "attendence_sys/facultyForm.html", context)
