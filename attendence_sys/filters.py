import django_filters

from .models import Attendence

class AttendenceFilter(django_filters.FilterSet):
    class Meta:
        model = Attendence
        fields = '__all__'
        exclude = ['time','created_at']