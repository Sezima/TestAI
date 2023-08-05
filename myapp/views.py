# myapp/views.py
from django.shortcuts import render, redirect
from .models import TextInfo
from rest_framework import viewsets
from rest_framework.viewsets import ModelViewSet

from .serializers import TextSerializer




class TextViewSet(ModelViewSet):
    queryset = TextInfo.objects.all()
    serializer_class = TextSerializer