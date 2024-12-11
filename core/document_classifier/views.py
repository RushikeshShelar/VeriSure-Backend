from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from .ml.classify import classify_document, preprocess_document

class DocumentClassifierView(APIView):
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

          # Ensure the directory exists
        upload_dir = "document_classifier/uploads"
        os.makedirs(upload_dir, exist_ok=True)


        # Save the file temporarily
        save_path = f"document_classifier/uploads/{file.name}"
        with open(save_path, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)

        try:
            document_class, confidence_score = classify_document(save_path)
            return Response({
                "class": document_class,
                "confidence_score": confidence_score,
                "file_path": save_path
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(save_path):
                os.remove(save_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
