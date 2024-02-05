from rest_framework import serializers
#
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(use_url=True)
