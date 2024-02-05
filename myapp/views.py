import cv2
import numpy as np
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ImageUploadSerializer
from tensorflow.keras.models import load_model


class PredictAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)

        if serializer.is_valid():
            image = serializer.validated_data['image']

            try:
                model = load_model(r'..\predict\eye_state.h5')
            except Exception as e:
                return Response({"error": "Failed to load the model."}, status=500)

            IMAGE_SIZE = 48

            try:
                img_array = np.frombuffer(image.read(), np.uint8)  # Updated from np.fromstring
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = img / 255.0
                img = img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

                prediction = model.predict(img)
                predicted_class = np.argmax(prediction[0])
                class_labels = ['no_yawn', 'yawn', 'Closed', 'Open']
            except Exception as e:
                return Response({"error": "Error processing image or prediction failed."}, status=500)

            return Response({"predicted_class": class_labels[predicted_class]})
        else:
            return Response({"errors": serializer.errors}, status=400)
