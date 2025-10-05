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
import re
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

business_data = [
    {
        "question": "ビジュアルアルファは何をする会社ですか？",
        "answer": "ビジュアルアルファは、東京を拠点とするB2Bフィンテックスタートアップで、機関投資家やアセットマネージャーのデータ運用を革新する包括的なSaaSデータソリューションを提供しています。当社のAI搭載プラットフォームは、複雑なデータ処理を自動化し、自動レポートを生成し、リアルタイムのポートフォリオモニタリング機能を提供します。非構造化金融データを実用的な洞察に変換し、投資チームの手作業によるExcel作業を最大80%削減することを専門としています。当社のソリューションは既存のシステムとシームレスに統合され、大規模な機関投資家のポートフォリオを管理するためのスケーラブルなインフラストラクチャを提供します。",
        "category": "company_overview",
        "image_path": "images/company_overview.png",
        "related_topics": ["サービス", "テクノロジー", "自動化"],
        "keywords": ["する", "会社", "ビジネス", "について", "概要", "サービス", "ビジュアル", "アルファ", "ビジュアルアルファ", "何をする", "何をする会社", "とは"]
    },
    {
        "question": "ビジュアルアルファの将来の目標は何ですか？",
        "answer": "ビジュアルアルファは、グローバル企業の顧客基盤を拡大し、データ駆動型ソリューションをより広範な国際的な顧客に提供することを目指しています。世界をリードする組織とのパートナーシップを通じて、グローバル金融エコシステムにおける存在感を強化し、機関投資家の進化するニーズに応えるためのイノベーションを続けていく計画です。",
        "category": "future_goals",
        "image_path": "images/future_goals.png",
        "related_topics": ["グローバル展開", "顧客", "成長", "国際"],
        "keywords": ["将来", "目標", "拡大", "グローバル", "計画", "ビジョン", "ロードマップ", "今後", "未来", "これから"]
    },
    {
        "question": "ビジュアルアルファはいつ設立されましたか？",
        "answer": "ビジュアルアルファは、金融セクターのデジタルトランスフォーメーションが著しい時期の2019年12月に東京で設立されました。設立以来、クライアントベースとテクノロジー機能の両面で急速な成長を遂げています。当社は、機関投資運用においてより効率的で透明性が高く、自動化されたソリューションを創造するというビジョンから生まれました。創業者たちは、投資運用とデータシステムにおける豊富な経験を活かし、日本市場における高度なフィンテックソリューションの需要の高まりに対応しています。",
        "category": "company_history",
        "image_path": "images/company_timeline.png",
        "related_topics": ["設立", "成長", "東京"],
        "keywords": ["設立", "いつ", "創立", "開始", "歴史", "2019", "起源", "創業", "始まった", "スタート"]
    },
    {
        "question": "ビジュアルアルファのチームの規模はどのくらいですか？",
        "answer": "2025年現在、ビジュアルアルファは取締役、技術顧問、コアスタッフを含め、15-20名程度の高度なスキルを持つプロフェッショナルを雇用しています。当社のチームは、フィンテック、データサイエンス、ソフトウェアエンジニアリング、金融サービスにわたる専門知識を持つ、多様で国際的な視野を持つ従業員で構成されています。世界有数の金融機関、テクノロジー企業、コンサルティング会社での経験を持つチームメンバーと共に、効率的な組織構造を維持しています。当社の文化は、イノベーション、継続的な学習、機関投資家クライアントへの卓越した価値提供を重視しています。",
        "category": "team_info",
        "image_path": "images/team_structure.png",
        "related_topics": ["スタッフ", "専門知識", "企業文化"],
        "keywords": ["チーム", "スタッフ", "従業員", "人員", "規模", "メンバー", "人数", "どのくらい", "何人", "社員"]
    },
    {
        "question": "ビジュアルアルファのリーダーは誰ですか？",
        "answer": "ビジュアルアルファは、投資管理データシステムと金融テクノロジーにおける豊富な経験を持つベテランプロフェッショナル、CEOのジェフリー・ツイが率いています。ジェフリーはビジュアルアルファを設立する前に、ステート・ストリート・コーポレーションやウェリントン・マネジメントなどの名門組織で働き、機関投資家がデータ管理とレポーティングにおいて直面する課題について深い洞察を得ました。彼のリーダーシップは、技術的専門知識と戦略的ビジョンを組み合わせ、日本及びそれ以上の地域で機関投資家による金融データの処理と活用を変革するという当社の使命を推進しています。",
        "category": "leadership",
        "image_path": "images/leadership_team.png",
        "related_topics": ["CEO", "経験", "背景"],
        "keywords": ["CEO", "リーダー", "創業者", "経営陣", "マネジメント", "ジェフリー", "ツイ", "誰", "率いている", "トップ", "代表"]
    },
    {
        "question": "ビジュアルアルファのクライアントには誰がいますか？",
        "answer": "ビジュアルアルファは、日本の主要なアセットマネージャーや年金基金を含む、名門機関投資家のポートフォリオにサービスを提供しています。注目すべきクライアントには、日本最大級の企業年金基金の一つであるベネッセグループ年金基金、大手資産運用会社の三井住友DSアセットマネジメント、コンサルティングサービスのグローバルリーダーであるマーサージャパンなどがあります。これらの関係は、大規模な機関投資家の厳格な要件を満たすエンタープライズグレードのソリューションを提供する当社の能力を示しており、複雑なポートフォリオ管理、規制報告、データ分析のニーズに対応しています。",
        "category": "clients",
        "image_path": "images/client_logos.png",
        "related_topics": ["機関投資家", "年金基金", "アセットマネージャー"],
        "keywords": ["クライアント", "顧客", "パートナー", "ベネッセ", "三井住友", "マーサー", "誰", "取引先", "お客様", "企業"]
    },
    {
        "question": "ビジュアルアルファはどのようなテクノロジーを使用していますか？",
        "answer": "ビジュアルアルファのテクノロジースタックは、高性能な金融データ処理のために設計された最新のスケーラブルなアーキテクチャに基づいて構築されています。バックエンドインフラストラクチャはNodeJSを使用してサーバーサイド開発を行い、高速で効率的なデータ処理を実現しています。フロントエンドはReactで構築され、レスポンシブで直感的なユーザーインターフェースを提供します。特定のWebアプリケーションにはLaravelフレームワークを使用したPHPを使用し、データベース層は信頼性の高いデータストレージのためにMySQLを採用しています。APIアーキテクチャには、柔軟なデータアクセスのためにGraphQLとRESTful APIの両方が含まれています。クラウドインフラストラクチャはAWSを通じて管理され、スケーラビリティとセキュリティを提供し、Dockerコンテナが一貫したデプロイメント環境を保証します。CI/CDパイプラインは、自動テストとデプロイメントのためにCircleCIを使用しています。",
        "category": "technology",
        "image_path": "images/tech_stack.png",
        "related_topics": ["nodejs", "react", "aws", "api"],
        "keywords": ["テクノロジー", "技術", "スタック", "ツール", "使用", "使っている", "使用している", "nodejs", "react", "aws", "docker", "データベース", "どのような", "どんな", "技術スタック", "テクノロジースタック", "どのようなテクノロジー", "テクノロジーを使用"]
    },
    {
        "question": "ビジュアルアルファの主なサービスは何ですか？",
        "answer": "ビジュアルアルファは、機関投資家向けに特別に設計された包括的な金融テクノロジーサービスのスイートを提供しています。当社のコアサービスには以下が含まれます：1) 非構造化データ処理 - 複雑な金融文書、レポート、データフィードを構造化された実用的な情報に変換、2) コンテンツ自動化 - 手作業の時間を節約する自動レポート、プレゼンテーション、分析文書の生成、3) パフォーマンス計算 - 正確でリアルタイムのポートフォリオパフォーマンスメトリクスとアトリビューション分析の提供、4) ポートフォリオモニタリング - 投資ポジション、リスクメトリクス、コンプライアンス要件の継続的な追跡。すべてのサービスは、大規模な機関投資を管理する金融専門家向けに調整され、既存の投資管理ワークフローとシームレスに統合されます。",
        "category": "services",
        "image_path": "images/services_overview.png",
        "related_topics": ["データ処理", "自動化", "ポートフォリオ管理"],
        "keywords": ["サービス", "提供", "製品", "ソリューション", "機能", "能力", "主な", "主要", "コア", "メイン"]
    },
    {
        "question": "マンデートを削除するにはどうすればよいですか？",
        "answer": "まず、マンデートを削除するには管理者アクセスが必要です。アプリケーションナビゲーションバーから、クライアント→マンデートを削除したいクライアントをクリック→クライアントのマンデート概要ページにリダイレクトされ、そのクライアントに関連付けられているすべてのマンデートのリストが表示されます→削除したいマンデートをクリック→マンデート詳細ページにリダイレクトされます→右上隅の「...」をクリック→開いたリストから「マンデートを削除」をクリックしてください。",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["マンデート", "削除", "管理者", "クライアント管理"],
        "keywords": ["削除", "マンデート", "除去", "方法", "管理者", "クライアント"]
    },
    {
        "question": "クライアントからマンデートを削除する手順は何ですか？",
        "answer": "マンデートを削除するには、管理者アクセスが必要です。ナビゲーションバーからクライアントに移動し、特定のクライアントを選択し、すべてのマンデートが表示されるマンデート概要ページを表示し、対象のマンデートをクリックしてマンデート詳細を開き、右上隅の「...」メニューをクリックして、ドロップダウンオプションから「マンデートを削除」を選択します。",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["マンデート", "削除", "管理者", "クライアント管理"],
        "keywords": ["除去", "マンデート", "削除", "手順", "クライアント", "管理者"]
    },
    {
        "question": "ビジュアルアルファでマンデートを削除するにはどうすればよいですか？",
        "answer": "マンデートの削除には管理者アクセスが必要です。アプリケーションナビゲーションバーから、クライアントをクリックし、マンデートを削除したいクライアントを選択します。クライアントのマンデート概要ページで、関連するすべてのマンデートが表示されます。削除したいマンデートをクリックしてマンデート詳細ページを開きます。次に、右上隅の「...」メニューボタンをクリックし、オプションから「マンデートを削除」を選択します。",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["マンデート", "削除", "管理者", "クライアント管理"],
        "keywords": ["削除", "マンデート", "ビジュアルアルファ", "方法", "管理者"]
    },
    {
        "question": "マンデートを削除するにはどのような権限が必要ですか？",
        "answer": "マンデートを削除するには管理者アクセスが必要です。管理者権限を持っている場合は、クライアント→クライアントを選択→マンデート概要ページを表示→特定のマンデートをクリック→マンデート詳細ページを開く→右上隅の「...」をクリック→ドロップダウンメニューから「マンデートを削除」を選択します。",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["マンデート", "削除", "管理者", "権限"],
        "keywords": ["権限", "管理者", "削除", "マンデート", "アクセス"]
    },
    {
        "question": "管理者アクセスなしでマンデートを削除できますか？",
        "answer": "いいえ、管理者アクセスなしではマンデートを削除できません。マンデートの削除には管理者アクセスが必要です。管理者アクセスをお持ちの場合は、次の手順に従ってください：ナビゲーションバーからクライアントをクリック→クライアントを選択→マンデート概要ページに移動→削除するマンデートをクリック→マンデート詳細を開く→右上の「...」をクリック→「マンデートを削除」を選択します。",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["マンデート", "削除", "管理者", "権限"],
        "keywords": ["管理者", "アクセス", "除去", "マンデート", "権限", "なし"]
    }
]

def create_sample_images():
    if not os.path.exists('images'):
        os.makedirs('images')
        print("📁 画像ディレクトリを作成しました")
    
    image_configs = [
        {'name': 'company_overview.png', 'text': 'Visual Alpha\nFintech Solutions', 'color': '#2563eb'},
        {'name': 'company_timeline.png', 'text': 'Founded 2019\nTokyo, Japan', 'color': '#059669'},
        {'name': 'team_structure.png', 'text': '15-20 Staff\nGlobal Team', 'color': '#dc2626'},
        {'name': 'leadership_team.png', 'text': 'Jeffrey Tsui\nCEO', 'color': '#7c3aed'},
        {'name': 'client_logos.png', 'text': 'Enterprise Clients\nInstitutional', 'color': '#ea580c'},
        {'name': 'tech_stack.png', 'text': 'NodeJS • React\nAWS • Docker', 'color': '#0891b2'},
        {'name': 'services_overview.png', 'text': 'Data Processing\nAutomation', 'color': '#be185d'},
        {'name': 'future_goals.png', 'text': 'Global Expansion\nGrowth', 'color': '#16a34a'}
    ]
    
    created_count = 0
    for config in image_configs:
        image_path = f"images/{config['name']}"
        if not os.path.exists(image_path):
            img = Image.new('RGB', (400, 200), color=config['color'])
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20) 
                except:
                    font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), config['text'], font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (400 - text_width) // 2
            y = (200 - text_height) // 2
            
            draw.text((x, y), config['text'], fill='white', font=font)
            img.save(image_path)
            created_count += 1
            print(f"✅ 作成しました: {image_path}")
    
    if created_count == 0:
        print("ℹ️  すべてのサンプル画像は既に存在します")
    else:
        print(f"🎨 {created_count}個のサンプル画像を作成しました")

class EnhancedBusinessKnowledgeBase:
    def __init__(self, data, embedding_model):
        self.data = data
        self.embedding_model = embedding_model
        self.index = None
        self.category_index = {}
        self.keyword_index = {}
        self.build_index()

    def build_index(self):
        # Build semantic embeddings
        texts = [f"{item['question']} {item['answer']} {' '.join(item.get('related_topics', []))}" 
                for item in self.data]
        embeddings = self.embedding_model.encode(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Build category index
        for i, item in enumerate(self.data):
            category = item.get('category', 'general')
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(i)
            
            # Build keyword index for fallback matching
            keywords = item.get('keywords', [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_index:
                    self.keyword_index[keyword_lower] = []
                self.keyword_index[keyword_lower].append(i)

    def keyword_match(self, query):
        """Fallback keyword matching for better recall"""
        query_lower = query.lower().strip()
        
        # Check for exact phrase matches first
        matches = {}
        for keyword in self.keyword_index:
            if keyword in query_lower:
                for idx in self.keyword_index[keyword]:
                    # Give higher weight to phrase matches
                    weight = len(keyword.split())
                    matches[idx] = matches.get(idx, 0) + weight
        
        # Also check individual words (for Japanese, check characters)
        query_words = re.findall(r'\w+', query_lower)
        for word in query_words:
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    matches[idx] = matches.get(idx, 0) + 0.5
        
        # Return indices sorted by match count
        if matches:
            sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
            return [idx for idx, count in sorted_matches if count >= 1]
        return []

    def search(self, query, top_k=2, min_score=0.30):
        """Hybrid search: semantic + keyword matching"""
        query_lower = query.lower().strip()
        
        # First try keyword matching for better precision on specific queries
        keyword_matches = self.keyword_match(query)
        
        # Debug: Print keyword matches
        if keyword_matches:
            print(f"🔍 Keyword matches found: {[self.data[idx]['question'][:50] for idx in keyword_matches[:3]]}")
        
        # Semantic search
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Debug: Print semantic matches
        print(f"🔍 Semantic matches: {[(self.data[idx]['question'][:50], float(scores[0][i])) for i, idx in enumerate(indices[0])]}")

        results = []
        
        # If we have a strong keyword match, prioritize it
        if keyword_matches:
            idx = keyword_matches[0]
            # Check if this is also in semantic results
            if idx in indices[0]:
                sem_idx = list(indices[0]).index(idx)
                score = float(scores[0][sem_idx])
            else:
                score = 0.6  # Give good score to keyword matches
            
            print(f"✅ Using keyword match: {self.data[idx]['question'][:60]}")
            
            results.append({
                'text': self.data[idx]['answer'],
                'score': score,
                'question': self.data[idx]['question'],
                'category': self.data[idx].get('category', 'general'),
                'image_path': self.data[idx].get('image_path'),
                'related_topics': self.data[idx].get('related_topics', []),
                'match_type': 'keyword'
            })
            return results
        
        # Otherwise use semantic results
        for idx, score in zip(indices[0], scores[0]):
            if score > min_score:
                results.append({
                    'text': self.data[idx]['answer'],
                    'score': float(score),
                    'question': self.data[idx]['question'],
                    'category': self.data[idx].get('category', 'general'),
                    'image_path': self.data[idx].get('image_path'),
                    'related_topics': self.data[idx].get('related_topics', []),
                    'match_type': 'semantic'
                })
        
        if results:
            print(f"✅ Using semantic match: {results[0]['question'][:60]}")
        
        return results

    def get_related_content(self, category, exclude_idx=None):
        if category in self.category_index:
            related = []
            for idx in self.category_index[category]:
                if exclude_idx is None or idx != exclude_idx:
                    related.append(self.data[idx])
            return related[:2]
        return []

    def get_response(self, query: str) -> Dict:
        """Legacy method for backward compatibility"""
        try:
            results = self.search(query, top_k=1, min_score=0.30)
            
            if not results:
                return {
                    'answer': 'すみません、その質問に対する回答が見つかりませんでした。',
                    'confidence': '低',
                    'distance': 999,
                    'related_topics': []
                }
            
            best_match = results[0]
            score = best_match['score']
            
            # Convert score to distance-like metric for backward compatibility
            distance = 1.0 - score if score > 0 else 999
            confidence = "高" if score > 0.5 else "中" if score > 0.35 else "低"
            
            return {
                'answer': best_match['text'],
                'confidence': confidence,
                'distance': distance,
                'related_topics': best_match.get('related_topics', []),
                'image_path': best_match.get('image_path'),
                'category': best_match.get('category', 'general')
            }
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
        self.kb = knowledge_base
        self.knowledge_base = knowledge_base  # For backward compatibility
        self.conversation_history = []
        
        # Configuration
        self.confidence_threshold = 0.30
        self.use_direct_answers = True
        self.max_response_length = 500

    def clean_response(self, response):
        """Clean and validate the response"""
        sentences = response.split('。')
        seen = set()
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen and len(sentence) > 10:
                seen.add(sentence)
                clean_sentences.append(sentence)
        
        cleaned = '。'.join(clean_sentences)
        if cleaned and not cleaned.endswith('。'):
            cleaned += '。'
        
        if len(cleaned) > self.max_response_length:
            cleaned = cleaned[:self.max_response_length].rsplit('。', 1)[0] + '。'
        
        return cleaned

    def validate_response(self, response, original_answer):
        """Check if response looks valid"""
        # Remove Japanese punctuation for word count
        words = re.findall(r'[\w]+', response)
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:
                return False
        
        if re.search(r'(.{20,}?)\1{2,}', response):
            return False
        
        if re.search(r'([。、；！？])\1{3,}', response):
            return False
        
        return True

    def generate_detailed_response(self, user_query, max_length=400) -> Tuple[str, Optional[str], List[str]]:
        user_query = user_query.strip()
        if not user_query:
            return self._handle_out_of_context("Empty query")
        
        relevant_docs = self.kb.search(user_query, top_k=2, min_score=self.confidence_threshold)
        
        if not relevant_docs:
            return self._handle_out_of_context(user_query)
        
        best_match = relevant_docs[0]
        response = best_match['text']
        
        if not self.validate_response(response, best_match['text']):
            response = best_match['text']
        
        response = self.clean_response(response)
        
        if len(response) < 50 or not self.validate_response(response, best_match['text']):
            response = best_match['text']
        
        image_path = best_match.get('image_path')
        related_topics = best_match.get('related_topics', [])
        
        self.conversation_history.append({
            'query': user_query,
            'response': response,
            'image_path': image_path,
            'topics': related_topics,
            'score': best_match['score'],
            'match_type': best_match.get('match_type', 'semantic')
        })
        
        return response, image_path, related_topics

    def _handle_out_of_context(self, user_query) -> Tuple[str, None, List[str]]:
        """Handle questions outside knowledge base"""
        response = """その情報は持っていません。今後の更新をお待ちいただくか、ナビゲーションバーから質問と回答を入力してください。

現在の知識に基づいて、ビジュアルアルファに関する以下の質問にのみ回答できます：
• 会社概要とサービス
• リーダーシップとチーム情報
• テクノロジースタックとソリューション
• クライアント情報
• 会社の歴史と設立
• 将来の目標と拡大計画

ビジュアルアルファに関連する質問をしてください。"""
        
        return response, None, []

    def get_response(self, query: str) -> Dict:
        """Main response method with backward compatibility"""
        try:
            print(f"Processing query in chatbot: {query}")
            
            response, image_path, related_topics = self.generate_detailed_response(query)
            
            image_base64 = None
            if image_path:
                try:
                    # Try relative path first
                    if os.path.exists(image_path):
                        img_path_to_use = image_path
                    else:
                        # Try parent directory path
                        img_path_to_use = os.path.join(os.path.dirname(os.path.dirname(__file__)), image_path)
                    
                    if os.path.exists(img_path_to_use):
                        with Image.open(img_path_to_use) as img:
                            if img.mode == 'RGBA':
                                img = img.convert('RGB')
                            
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='JPEG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                    else:
                        print(f"画像が見つかりません: {image_path}")
                except Exception as e:
                    print(f"画像の読み込みエラー: {str(e)}")

            # Determine confidence based on image availability and response content
            if image_path and image_base64:
                confidence = '高'
            elif "その情報は持っていません" in response:
                confidence = '低'
            else:
                confidence = '中'

            return {
                'response': response,
                'image_base64': image_base64,
                'confidence': confidence,
                'related_topics': related_topics
            }
        except Exception as e:
            print(f"Error in chatbot response: {str(e)}")
            return {
                'response': 'すみません、エラーが発生しました。',
                'image_base64': None,
                'confidence': '低',
                'related_topics': []
            }

def encode_image_to_base64(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"画像のエンコードエラー: {e}")
    return None

def create_placeholder_image(text="Visual Alpha", size=(400, 200)):
    img = Image.new('RGB', (400, 200), color='#1f2937')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    return img

def enhanced_chat_response(message: str) -> dict:
    """Main entry point for chat responses"""
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

def interactive_mode():
    print("\n🤖 インタラクティブチャットモード（終了するには'quit'と入力）")
    print("ビジュアルアルファについて何でも質問してください！")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👤 あなた: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '終了']:
                print("👋 さようなら！")
                break
                
            if not user_input:
                print("質問を入力してください！")
                continue
                
            print("🤖 処理中... ⏳")
            
            result = enhanced_chat_response(user_input)
            
            print(f"\n🤖 アシスタント:")
            print("-" * 40)
            print(result['response'])
            
            if result['image_base64']:
                timestamp = int(time.time())
                image_filename = f"chat_image_{timestamp}.png"
                image_data = base64.b64decode(result['image_base64'])
                with open(image_filename, 'wb') as f:
                    f.write(image_data)
                print(f"\n🖼️  関連画像を保存しました: {image_filename}")
            
            if result['related_topics']:
                print(f"\n🏷️  関連トピック: {', '.join(result['related_topics'])}")
            
            print(f"📊 信頼度: {result['confidence']}")
                
        except KeyboardInterrupt:
            print("\n👋 さようなら！")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

def run_tests():
    test_questions = [
        "ビジュアルアルファは何をする会社ですか？",
        "ビジュアルアルファとは",
        "テクノロジースタックについて教えてください", 
        "リーダーは誰ですか？",
        "主なサービスは何ですか？",
        "チームについて教えてください",
        "いつ設立されましたか？",
        "クライアントは誰ですか？",
        "今日の天気は？",
        "総理大臣は誰ですか？",
        "ピザの作り方は？",
    ]
    
    print("\n🧪 拡張チャットボットをテスト中:")
    print("=" * 50)
    
    total_tests = len(test_questions)
    passed_tests = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[テスト {i}/{total_tests}] 質問: {question}")
        
        try:
            start_time = time.time()
            result = enhanced_chat_response(question)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            print(f"回答 ({response_time:.1f}秒):")
            
            response_preview = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
            print(f"   {response_preview}")
            
            print(f"🖼️  画像: {'✅' if result['image_base64'] else '❌'}")
            print(f"🏷️  トピック: {', '.join(result['related_topics']) if result['related_topics'] else 'なし'}")
            print(f"📊 信頼度: {result['confidence']}")
            
            if i <= 8:
                is_valid = (
                    result['confidence'] in ['高', '中'] and
                    len(result['response'].strip()) > 50 and
                    "その情報は持っていません" not in result['response']
                )
                if is_valid:
                    passed_tests += 1
                    print("✅ テスト成功")
                else:
                    print("⚠️  テスト失敗")
            else:
                is_valid = (
                    result['confidence'] == '低' and 
                    "その情報は持っていません" in result['response']
                )
                if is_valid:
                    passed_tests += 1
                    print("✅ テスト成功（正しく拒否）")
                else:
                    print("⚠️  テスト失敗（拒否すべき）")
                
        except Exception as e:
            print(f"❌ エラー: {e}")
        
        print("-" * 50)
    
    print(f"\n📋 テスト概要:")
    print(f"   成功したテスト: {passed_tests}/{total_tests}")
    print(f"   成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 すべてのテストが成功しました！")
    elif passed_tests >= total_tests * 0.8:
        print("✅ ほとんどのテストが成功しました。")
    else:
        print("⚠️  いくつかのテストが失敗しました。")

if __name__ == "__main__":
    try:
        print("\n🤖 チャットボットを初期化中...")
        create_sample_images()
        kb = EnhancedBusinessKnowledgeBase(business_data, embedding_model)
        chatbot = EnhancedBusinessChatbot(model, tokenizer, kb)
        print("✅ チャットの準備ができました！")
        
        # コメントを外して使用してください:
        interactive_mode()
        # run_tests()
    except KeyboardInterrupt:
        print("\n👋 さようなら！")
    except Exception as e:
        print(f"\n❌ エラー: {e}")