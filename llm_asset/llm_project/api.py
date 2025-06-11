import os
import gradio as gr
import requests
import re
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import kss
from textblob import TextBlob
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import google.generativeai as genai

#gemini 호출 함수 작성
genai.configure(api_key="AIzaSyCQIYftXpxtC-LhNF9f56p4AOgT8qfpjN8")


def ask_gemini_for_route(current_location: str, destination_info: str)-> str:
    prompt = f"""
    현재 위치: {current_location}
    목적지: {destination_info}
    
    현재 위치에서 목적지 까지 대중교통 또는 도보로 이동하는 가장 효율적인 방법을 설명해주세요. 
    경로에 대한 설명은 한국어로 자세하게 풀어주세요.
    """
    
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content(prompt)
    return response.text
