import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .utils.eval_functions import eval_prod_with_one_prop, eval_prod_with_two_prop, eval_prod_with_three_prop


# Create your views here.
@csrf_exempt
def evaluate_product_with_one_property(request):
    if request.method == 'POST':
        print("evaluate product with one property function called")
        data = json.loads(request.body)

        evaluated_price, early_redempted_probabilities, final_gain_prob, loss_prob = eval_prod_with_one_prop(data)
        response_data = {'evaluated_price': evaluated_price,
                         'early_redempted_probabilities': early_redempted_probabilities,
                         'final_gain_prob': final_gain_prob, 'loss_prob': loss_prob}

        response = JsonResponse(response_data)
        response["Access-Control-Allow-Origin"] = "*"  # 모든 출처 허용
        response["Access-Control-Allow-Methods"] = "POST"  # 허용할 메서드 설정

        return response


@csrf_exempt
def evaluate_product_with_two_properties(request):
    if request.method == 'POST':
        print("evaluate product with two properties function called")
        data = json.loads(request.body)

        evaluated_price, early_redempted_probabilities, final_gain_prob, loss_prob = eval_prod_with_two_prop(data)

        response_data = {'evaluated_price': evaluated_price,
                         'early_redempted_probabilities': early_redempted_probabilities,
                         'final_gain_prob': final_gain_prob, 'loss_prob': loss_prob}

        response = JsonResponse(response_data)
        response["Access-Control-Allow-Origin"] = "*"  # 모든 출처 허용
        response["Access-Control-Allow-Methods"] = "POST"  # 허용할 메서드 설정

        return response


@csrf_exempt
def evaluate_product_with_three_properties(request):
    if request.method == 'POST':
        print("evaluate product with three properties function called")
        data = json.loads(request.body)

        evaluated_price, early_redempted_probabilities, final_gain_prob, loss_prob = eval_prod_with_three_prop(data)

        response_data = {'evaluated_price': evaluated_price,
                         'early_redempted_probabilities': early_redempted_probabilities,
                         'final_gain_prob': final_gain_prob, 'loss_prob': loss_prob}

        response = JsonResponse(response_data)
        response["Access-Control-Allow-Origin"] = "*"  # 모든 출처 허용
        response["Access-Control-Allow-Methods"] = "POST"  # 허용할 메서드 설정

        return response





