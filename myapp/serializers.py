from rest_framework import serializers

from  .models import *

class TextSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextInfo
        fields = '__all__'


