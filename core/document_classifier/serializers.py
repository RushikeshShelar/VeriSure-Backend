from rest_framework import serializers

class DocumentClassifierSerializer(serializers.Serializer):
    file = serializers.FileField()