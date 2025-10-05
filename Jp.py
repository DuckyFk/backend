import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import os
import time
from typing import List, Dict, Tuple, Optional

print("🔄 日本語モデルを読み込んでいます... (初回実行時は数分かかる場合があります)")

model_name = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
kb = None
chatbot = None

print("✅ モデルの読み込みが完了しました!")

def initialize_bot():
    global kb, chatbot
    kb = EnhancedBusinessKnowledgeBase(business_data, embedding_model)
    chatbot = EnhancedBusinessChatbot(model, tokenizer, kb)

def enhanced_chat_response(message: str) -> dict:
    if chatbot is None:
        return {
            "response": "チャットボットが初期化されていません。",
            "image_base64": None,
            "confidence": "低",
            "related_topics": []
        }
    
    try:
        result = chatbot.get_response(message)
        return {
            "response": result['response'],
            "image_base64": result.get('image_base64'),
            "confidence": result.get('confidence', '低'),
            "related_topics": result.get('related_topics', [])
        }
    except Exception as e:
        print(f"Error in enhanced_chat_response: {str(e)}")
        return {
            "response": "申し訳ありませんが、エラーが発生しました。",
            "image_base64": None,
            "confidence": "低",
            "related_topics": []
        }

business_data = [
    {
        "question": "ビジュアルアルファは何をする会社ですか？",
        "answer": "ビジュアルアルファは、東京を拠点とするB2Bフィンテックスタートアップで、機関投資家やアセットマネージャーのデータ運用を革新する包括的なSaaSデータソリューションを提供しています。当社のAI搭載プラットフォームは、複雑なデータ処理を自動化し、自動レポートを生成し、リアルタイムのポートフォリオモニタリング機能を提供します。非構造化金融データを実用的な洞察に変換し、投資チームの手作業によるExcel作業を最大80%削減することを専門としています。当社のソリューションは既存のシステムとシームレスに統合され、大規模な機関投資家のポートフォリオを管理するためのスケーラブルなインフラストラクチャを提供します。",
        "category": "company_overview",
        "image_path": "images/company_overview.png",
        "related_topics": ["サービス", "テクノロジー", "自動化"]
    },
    {
        "question": "ビジュアルアルファはいつ設立されましたか？",
        "answer": "ビジュアルアルファは、金融セクターのデジタルトランスフォーメーションが著しい時期の2019年12月に東京で設立されました。設立以来、クライアントベースとテクノロジー機能の両面で急速な成長を遂げています。当社は、機関投資運用においてより効率的で透明性が高く、自動化されたソリューションを創造するというビジョンから生まれました。創業者たちは、投資運用とデータシステムにおける豊富な経験を活かし、日本市場における高度なフィンテックソリューションの需要の高まりに対応しています。",
        "category": "company_history",
        "image_path": "images/company_timeline.png",
        "related_topics": ["設立", "成長", "東京"]
    },
    {
        "question": "ビジュアルアルファのチームの規模はどのくらいですか？",
        "answer": "2025年現在、ビジュアルアルファは取締役、技術顧問、コアスタッフを含め、15-20名程度の高度なスキルを持つプロフェッショナルを雇用しています。当社のチームは、フィンテック、データサイエンス、ソフトウェアエンジニアリング、金融サービスにわたる専門知識を持つ、多様で国際的な視野を持つ従業員で構成されています。世界有数の金融機関、テクノロジー企業、コンサルティング会社での経験を持つチームメンバーと共に、効率的な組織構造を維持しています。当社の文化は、イノベーション、継続的な学習、機関投資家クライアントへの卓越した価値提供を重視しています。",
        "category": "team_info",
        "image_path": "images/team_structure.png",
        "related_topics": ["スタッフ", "専門知識", "企業文化"]
    }
]

class EnhancedBusinessKnowledgeBase:
    def __init__(self, data: List[Dict], embedding_model):
        self.data = data
        self.embedding_model = embedding_model
        self.index = None
        self._build_index()

    def _build_index(self):
        questions = [item["question"] for item in self.data]
        embeddings = self.embedding_model.encode(questions)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def get_most_similar(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
        return list(zip(I[0], D[0]))

    def get_response(self, query: str) -> Dict:
        try:
            similar_questions = self.get_most_similar(query, k=1)
            best_match_idx, distance = similar_questions[0]
            
            if best_match_idx >= len(self.data):
                return {
                    'answer': 'すみません、その質問に対する回答が見つかりませんでした。',
                    'confidence': '低',
                    'distance': 999,
                    'related_topics': []
                }
            
            confidence = "高" if distance < 1.5 else "中" if distance < 2.0 else "低"
            
            response_data = self.data[best_match_idx].copy()
            response_data['confidence'] = confidence
            response_data['distance'] = distance
            
            return response_data
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return {
                'answer': 'すみません、エラーが発生しました。',
                'confidence': '低',
                'distance': 999,
                'related_topics': []
            }

class EnhancedBusinessChatbot:
    def __init__(self, model, tokenizer, knowledge_base):
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge_base = knowledge_base

    def get_response(self, query: str) -> Dict:
        try:
            print(f"Processing query in chatbot: {query}")
            response_data = self.knowledge_base.get_response(query)
            print(f"Knowledge base response: {response_data}")
            
            image_base64 = None
            if response_data.get('image_path'):
                try:
                    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), response_data['image_path'])
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            if img.mode == 'RGBA':
                                img = img.convert('RGB')
                            
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='JPEG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                except Exception as e:
                    print(f"画像の読み込みエラー: {str(e)}")

            return {
                'response': response_data.get('answer', 'エラーが発生しました'),
                'image_base64': image_base64,
                'confidence': response_data.get('confidence', '低'),
                'related_topics': response_data.get('related_topics', [])
            }
        except Exception as e:
            print(f"Error in chatbot response: {str(e)}")
            return {
                'response': 'すみません、エラーが発生しました。',
                'image_base64': None,
                'confidence': '低',
                'related_topics': []
            }