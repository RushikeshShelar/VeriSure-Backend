from rest_framework import serializers

from .models import Application

class ApplicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Application
        fields = ['id', 'name', 'username', 'data', 'required_documents', 'created_at']