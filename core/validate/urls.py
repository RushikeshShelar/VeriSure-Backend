from django.urls import path
from .views import VerifyView, OCRLabelingBatchView

urlpatterns = [
    path('', VerifyView.as_view(), name='verify'),
    path('bulk/', OCRLabelingBatchView.as_view(), name='final-bulk0.'),
]