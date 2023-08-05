# myapp/views.py
from django.shortcuts import render, redirect
from .models import TextInfo
from rest_framework import viewsets
from rest_framework.viewsets import ModelViewSet
from drf_yasg.utils import swagger_auto_schema
from .serializers import TextSerializer
from drf_yasg import openapi
from transformers import AdamW
from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextViewSet(ModelViewSet):
    queryset = TextInfo.objects.all()
    serializer_class = TextSerializer



# Создание своего датасета с вопросами и ответами
data = [
    {
        "question": "Как открыть банковский счет?",
        "answer": "Чтобы открыть банковский счет, вам необходимо обратиться в ближайшее отделение банка, предоставить необходимые документы и заполнить заявление на открытие счета."
    },
    {
        "question": "Что делать, если забыл пин-код от банковской карты?",
        "answer": "Если вы забыли пин-код от банковской карты, вам следует обратиться в банк с паспортом и заявлением на сброс пин-кода."
    },
    # Добавьте больше вопросов и ответов
]

# Создание экземпляра токенизатора и модели GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = GPT2LMHeadModel.from_pretrained(model_name)

# Форматируем данные для обучения
texts = [item["question"] + " " + item["answer"] for item in data]

# Токенизируем данные
input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").input_ids

# Обучение модели на своих данных
optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()

for epoch in range(3):  # Выполним 3 эпохи обучения (можете увеличить или уменьшить число эпох в зависимости от данных)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Функция для генерации ответов
def generate_answer(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        model.eval()
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type='object',
    properties={
        'input_text': openapi.Schema(type='string')
    }
))
@api_view(['POST'])

def generate_response(request):
    if request.method == 'POST':
        input_text = request.data.get('input_text')

        # Check if input_text is provided
        if not input_text:
            return Response({'error': 'Input text is required.'}, status=400)

        # Загрузите предварительно обученную модель GPT-2 и токенизатор
        model_name = "gpt2"  # Можно использовать другие модели, например, "gpt2-medium", "gpt2-large", "gpt2-xl"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Преобразование входного текста в токены и добавление специальных токенов
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Генерация ответа на основе входных токенов
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return Response({'generated_text': generated_text})

    return Response({'error': 'Invalid request method.'}, status=405)