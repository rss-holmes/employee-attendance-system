from django.shortcuts import render, redirect
from django.http import HttpResponse, StreamingHttpResponse

from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .forms import *
from .models import Employee, Attendence
from .filters import AttendenceFilter
from django.db.models import Q
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# from django.views.decorators import gzip

from .recognizer import Recognizer
from datetime import date

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


@csrf_exempt
@login_required(login_url="login")
def takeAttendence(request):
    if request.method == "POST":
        image = request.FILES['webcam']
        temp_file_path = default_storage.save('temp/c1.jpg', ContentFile(image.read()))
        # tmp_file = os.path.join(settings.MEDIA_ROOT, path)

        names = Recognizer(temp_file_path)

        company_employees = Employee.objects.all()
        for employee in company_employees:
            if str(employee.employee_id) in names:
                attendance = Attendence(
                        entry_by=request.user.username,
                        employee_id=employee.employee_id,
                        employee_name=employee.firstname + ' ' + employee.lastname,
                        status="Present",
                    )
                attendance.save()

        todays_attendences = Attendence.objects.filter(
            date=str(date.today())
        )
        context = {"attendences": todays_attendences, "ta": True}
        if len(names) == 0:
            messages.error(request, "No employee could be identified in the capture.")
        else:
            messages.success(request, "Attendence taking Success.Employees identified :: " + str(names))
        return render(request, "attendence_sys/attendence.html", context)
    context = {}
    return render(request, "attendence_sys/home.html", context)

def mark_attendance(request, company_employees):

    attendances = []
    for employee in company_employees:
        attendance = Attendence(
                        entry_by=request.user.username,
                        employee_id=employee.employee_id,
                        employee_name=employee.firstname + ' ' + employee.lastname,
                        status="Present",
                    )

        attendance.save()
        attendances.append(attendance)

    return attendances

@login_required(login_url="login")
def takeManualAttendence(request):
    if request.method == "POST":
        details = {
            "rfid": request.POST["rfid"],
            "employee_id": request.POST["employee_id"],
        }

        if not details.get('rfid', None) and not details.get('employee_id', None) :
            messages.error(request, "No employees details were entered by the user.")
            return render(request, "attendence_sys/attendence.html", {})

        q_object = Q()
        q_object = q_object & Q(rfid=details["rfid"]) if details["rfid"] else q_object
        q_object = q_object & Q(employee_id=details["employee_id"]) if details["employee_id"] else q_object

        company_employees = Employee.objects.filter(q_object)

        if len(company_employees) == 0:
            messages.error(request, "No employees could be found matching the details")
            return render(request, "attendence_sys/attendence.html", {})

        attendences = mark_attendance(request, company_employees)
        messages.success(request, "Attendence taking Success")
        return render(request, "attendence_sys/attendence.html", {"attendences": attendences, "ta": True})

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
