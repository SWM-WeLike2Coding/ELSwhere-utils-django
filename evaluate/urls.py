from django.urls import path

from . import views
from .views import evaluate_product_with_one_property, evaluate_product_with_two_properties, evaluate_product_with_three_properties


app_name = 'chatbot-namespace'


urlpatterns = [
    path("evaluate-product-with-one-property/", evaluate_product_with_one_property, name='evaluate_product_with_one_property'),
    path("evaluate-product-with-two-properties/", evaluate_product_with_two_properties, name='evaluate_product_with_two_properties'),
    path("evaluate-product-with-three-properties/", evaluate_product_with_three_properties, name='evaluate_product_with_three_properties'),

]