from django.urls import path
from .views import DocumentClassifierView

urlpatterns = [
    path('classify/', DocumentClassifierView.as_view(), name='classify')
]