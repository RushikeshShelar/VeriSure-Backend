import os
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PyPDF2 import PdfReader

from document_classifier.ml.classify import classify_document
from ocr_labeling.index import run_ocr, run_ollama
from validate.llm import run_openai

from pdf2image import convert_from_path
import os


class VerifyView(APIView):
    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "Invalid Inputs"}, status=status.HTTP_400_BAD_REQUEST)

          # Ensure the directory exists
        upload_dir = "verified/"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file temporarily
        save_path = f"verified/{file.name}"
        with open(save_path, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)
                
        try:
            document_class, confidence_score = classify_document(save_path)
            if float(confidence_score) < 0.75:
                if os.path.exists(save_path):
                    os.remove(save_path)
                return Response({
                    "class": "Invalid",
                    "file_path": save_path
                }, status=status.HTTP_200_OK)
                
            ocr_data = run_ocr(save_path)
            ollama_output = run_ollama(ocr_data)
            
            print(ollama_output)
            
            # actual_input = Application.objects.get(id=id)
            
            return Response({
                "class": document_class,
                "ocr_data": ocr_data,
                "ollama_output": ollama_output,
                "file_path": save_path
            }, status=status.HTTP_200_OK)
            
            
        
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(save_path):
                os.remove(save_path)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



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