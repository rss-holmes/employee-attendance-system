# Generated by Django 3.0.8 on 2020-11-16 14:48

import attendence_sys.models
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Attendence',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('entry_by', models.CharField(blank=True, max_length=200, null=True)),
                ('employee_id', models.CharField(blank=True, max_length=200, null=True)),
                ('employee_name', models.CharField(blank=True, max_length=200, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('date', models.DateField(auto_now_add=True, null=True)),
                ('time', models.TimeField(auto_now_add=True, null=True)),
                ('status', models.CharField(default='Absent', max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Employee',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('firstname', models.CharField(blank=True, max_length=200, null=True)),
                ('lastname', models.CharField(blank=True, max_length=200, null=True)),
                ('employee_id', models.CharField(max_length=200, null=True)),
                ('company_name', models.CharField(blank=True, max_length=200, null=True)),
                ('temperature', models.FloatField()),
                ('spo2', models.FloatField()),
                ('entry_by', models.CharField(blank=True, max_length=200, null=True)),
                ('profile_pic', models.ImageField(blank=True, null=True, upload_to=attendence_sys.models.employee_directory_path)),
            ],
        ),
        migrations.CreateModel(
            name='Faculty',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('firstname', models.CharField(blank=True, max_length=200, null=True)),
                ('lastname', models.CharField(blank=True, max_length=200, null=True)),
                ('phone', models.CharField(max_length=200, null=True)),
                ('email', models.CharField(max_length=200, null=True)),
                ('profile_pic', models.ImageField(blank=True, null=True, upload_to=attendence_sys.models.user_directory_path)),
                ('user', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
