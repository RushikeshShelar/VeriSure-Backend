import os
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PyPDF2 import PdfReader

from document_classifier.ml.classify import classify_document
from ocr_labeling.index import run_ocr, run_ollama
from validate.llm import run_openai
# from models.index import run_ocr


from pdf2image import convert_from_path
import os


class VerifyView(APIView):
    def post(self, request):
        file = request.FILES.get('file')  # Get the single file uploaded with the 'file' key
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Define upload directory based on file type
            upload_dir = "media/pdf/" if file.name.endswith('.pdf') else "media/images/"
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, file.name)

            # Save the file temporarily
            with open(save_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            if file.name.endswith('.pdf'):
                # Convert PDF to a single image
                image_path = os.path.join("media/images/", f"{os.path.splitext(file.name)[0]}.jpg")
                images = convert_from_path(save_path, dpi=300) 
                images[0].save(image_path, 'JPEG')

                # Process the image (example: classify and run OCR)
                document_class, confidence_score = classify_document(image_path)
                ocr_text = run_ocr(image_path)

                return Response({
                    "file_name": file.name,
                    "class": document_class,
                    "confidence_score": confidence_score,
                    "ocr_text": ocr_text,
                    "file_path": image_path
                }, status=status.HTTP_200_OK)

            else:
                # Process non-PDF files directly
                document_class, confidence_score = classify_document(save_path)
                ocr_text = run_ocr(save_path)

                return Response({
                    "file_name": file.name,
                    "class": document_class,
                    "confidence_score": confidence_score,
                    "ocr_text": ocr_text,
                    "file_path": save_path
                }, status=status.HTTP_200_OK)

        except Exception as e:
            # Handle any errors that occur during processing
            return Response({
                "file_name": file.name,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
      
            
            
            



class OCRLabelingBatchView(APIView):
    def post(self, request):
        files = request.FILES.getlist('files')  # Get all files uploaded with the 'files' key
        if not files:
            return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        start_time = time.time()

        for file in files:
            try:
                # Define upload directory based on file type
                upload_dir = "media/pdf/" if file.name.endswith('.pdf') else "media/images/"
                os.makedirs(upload_dir, exist_ok=True)
                save_path = os.path.join(upload_dir, file.name)

                # Save the file temporarily
                with open(save_path, 'wb') as f:
                    for chunk in file.chunks():
                        f.write(chunk)

                if file.name.endswith('.pdf'):
                    # Convert PDF to images
                    image_dir = os.path.join("media/images/", os.path.splitext(file.name)[0])
                    os.makedirs(image_dir, exist_ok=True)
                    images = convert_from_path(save_path, dpi=300)  # Adjust DPI as needed

                    for _, page in enumerate(images):
                        image_path = os.path.join(image_dir, f"page_{file}.jpg")
                        page.save(image_path, 'JPEG')

                        # Process the image (example: classify and run OCR)
                        document_class, confidence_score = classify_document(image_path)
                        ocr_text = run_ocr(image_path)

                        results.append({
                            "file_name": file.name,
                            "page": os.path.basename(image_path),
                            "class": document_class,
                            "confidence_score": confidence_score,
                            "ocr_text": ocr_text,
                            "file_path": image_path
                        })
                else:
                    # Process non-PDF files directly
                    document_class, confidence_score = classify_document(save_path)
                    ocr_text = run_ocr(save_path)

                    results.append({
                        "file_name": file.name,
                        "class": document_class,
                        "confidence_score": confidence_score,
                        "ocr_text": ocr_text,
                        "file_path": save_path
                    })

            except Exception as e:
                # Handle errors for individual files
                results.append({
                    "file_name": file.name,
                    "error": str(e)
                })

        end_time = time.time()
        processing_time = end_time - start_time

        return Response({"results": results, "processing_time": processing_time}, status=status.HTTP_200_OK)