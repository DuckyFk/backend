import os
import re
import io
import time
import base64
import torch
import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

print("🔄 Loading models... (this may take a few minutes on first run)")

# Load models
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

kb = None
chatbot = None

print("✅ Models loaded successfully!")


def initialize_bot():
    """Initialize chatbot and knowledge base."""
    global kb, chatbot
    kb = EnhancedBusinessKnowledgeBase(business_data, embedding_model)
    chatbot = EnhancedBusinessChatbot(model, tokenizer, kb)


# ---------------------------
# Business Data
# ---------------------------
business_data = [
    {
        "question": "What does Visual Alpha do?",
        "answer": "Visual Alpha is a Tokyo-based B2B fintech startup offering comprehensive SaaS data solutions that revolutionize how institutional investors and asset managers handle their data operations. Our AI-powered platform automates complex data processing, generates automated reports, and provides real-time portfolio monitoring capabilities. We specialize in transforming unstructured financial data into actionable insights, reducing manual Excel work by up to 80% for investment teams. Our solutions integrate seamlessly with existing systems and provide scalable infrastructure for managing large-scale institutional portfolios.",
        "category": "company_overview",
        "image_path": "images/company_overview.png",
        "related_topics": ["services", "technology", "automation"],
        "keywords": ["do", "does", "company", "business", "about", "overview", "services", "visual", "alpha", "fintech"], 
    },
    {
        "question": "What are Visual Alpha's future goals?",
        "answer": "Visual Alpha aims to expand its client base among global firms, bringing its data-driven solutions to a broader international audience. By partnering with leading organizations worldwide, the company plans to strengthen its presence in the global financial ecosystem and continue innovating to meet the evolving needs of institutional clients.",
        "category": "future_goals",
        "image_path": "images/future_goals.png",
        "related_topics": ["global expansion", "clients", "growth", "international"],
        "keywords": ["future", "goals", "expansion", "global", "plans", "vision", "roadmap"]
    },
    {
        "question": "When was Visual Alpha founded?",
        "answer": "Visual Alpha was established in December 2019 in Tokyo, Japan, during a period of significant digital transformation in the financial sector. Since our founding, we have experienced rapid growth, expanding our client base and technology capabilities. The company was born from the vision of creating more efficient, transparent, and automated solutions for institutional investment management. Our founders leveraged their extensive experience in investment management and data systems to address the growing need for sophisticated fintech solutions in the Japanese market.",
        "category": "company_history",
        "image_path": "images/company_timeline.png",
        "related_topics": ["founding", "growth", "tokyo"],
        "keywords": ["founded", "when", "established", "started", "history", "2019", "origin"]
    },
    {
        "question": "How large is the Visual Alpha team?",
        "answer": "As of 2025, Visual Alpha employs around 15-20 highly skilled professionals, including board directors, technical advisors, and core staff members. Our team represents a diverse, internationally-minded workforce with expertise spanning fintech, data science, software engineering, and financial services. We maintain a lean but highly effective organizational structure, with team members bringing experience from leading global financial institutions, technology companies, and consulting firms. Our collaborative culture emphasizes innovation, continuous learning, and delivering exceptional value to our institutional clients.",
        "category": "team_info",
        "image_path": "images/team_structure.png",
        "related_topics": ["staff", "expertise", "culture"],
        "keywords": ["team", "staff", "employees", "people", "size", "members", "workforce"]
    },
    {
        "question": "Who leads Visual Alpha?",
        "answer": "Visual Alpha is led by CEO Jeffrey Tsui, a seasoned professional with extensive experience in investment management data systems and financial technology. Before founding Visual Alpha, Jeffrey worked with prestigious organizations including State Street Corporation and Wellington Management, where he gained deep insights into the challenges faced by institutional investors in data management and reporting. His leadership combines technical expertise with strategic vision, driving the company's mission to transform how financial data is processed and utilized by institutional investors across Japan and beyond.",
        "category": "leadership",
        "image_path": "images/leadership_team.png",
        "related_topics": ["ceo", "experience", "background"],
        "keywords": ["ceo", "leader", "founder", "executive", "management", "jeffrey", "tsui", "who"]
    },
    {
        "question": "Who are some of Visual Alpha's clients?",
        "answer": "Visual Alpha serves a prestigious portfolio of institutional clients, including leading asset managers and pension funds in Japan. Our notable clients include Benesse Group Pension Fund, one of Japan's largest corporate pension funds; Sumitomo Mitsui DS Asset Management, a major asset management company; and Mercer Japan, a global leader in consulting services. These relationships demonstrate our ability to deliver enterprise-grade solutions that meet the stringent requirements of large-scale institutional investors, handling complex portfolio management, regulatory reporting, and data analytics needs.",
        "category": "clients",
        "image_path": "images/client_logos.png",
        "related_topics": ["institutional_investors", "pension_funds", "asset_managers"],
        "keywords": ["clients", "customers", "partners", "benesse", "sumitomo", "mercer", "who"]
    },
    {
        "question": "What technologies does Visual Alpha use?",
        "answer": "Visual Alpha's technology stack is built on modern, scalable architecture designed for high-performance financial data processing. Our backend infrastructure utilizes NodeJS for server-side development, ensuring fast and efficient data processing. The frontend is built with React, providing responsive and intuitive user interfaces. We use PHP with the Laravel framework for certain web applications, while our database layer is powered by MySQL for reliable data storage. Our API architecture includes both GraphQL and RESTful APIs for flexible data access. Cloud infrastructure is managed through AWS, providing scalability and security, while Docker containers ensure consistent deployment environments. Our CI/CD pipeline is powered by CircleCI for automated testing and deployment.",
        "category": "technology",
        "image_path": "images/tech_stack.png",
        "related_topics": ["nodejs", "react", "aws", "api"],
        "keywords": ["technology", "tech", "stack", "tools", "nodejs", "react", "aws", "docker", "database"]
    },
    {
        "question": "What are Visual Alpha's main services?",
        "answer": "Visual Alpha offers a comprehensive suite of financial technology services designed specifically for institutional investors. Our core services include: 1) Unstructured Data Processing - transforming complex financial documents, reports, and data feeds into structured, actionable information; 2) Content Automation - generating automated reports, presentations, and analytical documents that save hours of manual work; 3) Performance Calculation - providing accurate, real-time portfolio performance metrics and attribution analysis; 4) Portfolio Monitoring - continuous tracking of investment positions, risk metrics, and compliance requirements. All services are tailored for financial professionals managing large-scale institutional investments and integrate seamlessly with existing investment management workflows.",
        "category": "services",
        "image_path": "images/services_overview.png",
        "related_topics": ["data_processing", "automation", "portfolio_management"],
        "keywords": ["services", "offerings", "products", "solutions", "features", "capabilities"]
    },
    {
        "question": "How do I delete a mandate?",
        "answer": "First of all, you will need admin access to delete a mandate. From the application navbar, click on Clients → then click on the client from which you want to delete the mandate → you will be redirected to the client's Mandate Summary page where you can see all the listed mandates associated with that client → then click on the mandate you want to delete → you will be redirected to the Mandate Details page → then click on '...' on the top right corner → from the list that opens, click on 'Delete mandate'.",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["mandate", "delete", "admin", "client management"],
        "keywords": ["delete", "mandate", "remove", "how", "admin", "client"]
    },
    {
        "question": "What are the steps to remove a mandate from a client?",
        "answer": "To remove a mandate, you must have admin access. Navigate to Clients from the navbar, select the specific client, view their Mandate Summary page showing all mandates, click on the target mandate to open Mandate Details, then click the '...' menu in the top right corner and select 'Delete mandate' from the dropdown options.",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["mandate", "delete", "admin", "client management"],
        "keywords": ["remove", "mandate", "delete", "steps", "client", "admin"]
    },
    {
        "question": "How can I delete a mandate in Visual Alpha?",
        "answer": "Deleting a mandate requires admin access. From the application navbar, click on Clients, then select the client whose mandate you want to delete. On the client's Mandate Summary page, you'll see all associated mandates. Click on the mandate you wish to delete to open its Mandate Details page. Then click on the '...' menu button on the top right corner and select 'Delete mandate' from the options.",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["mandate", "delete", "admin", "client management"],
        "keywords": ["delete", "mandate", "visual alpha", "how", "admin"]
    },
    {
        "question": "What permissions do I need to delete a mandate?",
        "answer": "You need admin access to delete a mandate. Once you have admin permissions, navigate to Clients → select the client → view the Mandate Summary page → click on the specific mandate → open the Mandate Details page → click '...' in the top right corner → select 'Delete mandate' from the dropdown menu.",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["mandate", "delete", "admin", "permissions"],
        "keywords": ["permissions", "admin", "delete", "mandate", "access"]
    },
    {
        "question": "Can I remove a mandate without admin access?",
        "answer": "No, you cannot delete a mandate without admin access. Admin access is required to perform mandate deletion. If you have admin access, follow these steps: From the navbar, click Clients → select the client → go to Mandate Summary page → click the mandate to delete → open Mandate Details → click '...' on the top right → select 'Delete mandate'.",
        "category": "mandate_management",
        "image_path": "images/delete_mandate.png",
        "related_topics": ["mandate", "delete", "admin", "permissions"],
        "keywords": ["admin", "access", "remove", "mandate", "permissions", "without"]
    },
        {
        "question": "How do I add a new client in Visual Alpha?",
        "answer": "To add a new client, you need admin access. From the navbar, click on Clients → then click 'Add Client' → fill in the client details such as name, contact information, and relevant documents → click 'Save' to register the client in the system.",
        "category": "client_management",
        "image_path": "images/add_client.png",
        "related_topics": ["clients", "admin", "add", "registration"],
        "keywords": ["add", "client", "register", "new", "how", "create"],
        "active": True
    },
    {
        "question": "How do I update client information?",
        "answer": "Navigate to Clients from the navbar → select the client you want to update → open the Client Details page → click 'Edit' → update the necessary information → click 'Save' to apply the changes.",
        "category": "client_management",
        "image_path": "images/edit_client.png",
        "related_topics": ["clients", "update", "edit"],
        "keywords": ["update", "edit", "client", "information", "how", "change"],
        "active": True
    },
    {
        "question": "How can I generate a report in Visual Alpha?",
        "answer": "Go to the Reports section in the application → select the type of report you need → choose the relevant client or portfolio → apply any filters if necessary → click 'Generate' → the report will be displayed and can be exported as PDF or Excel.",
        "category": "reporting",
        "image_path": "images/generate_report.png",
        "related_topics": ["reports", "export", "pdf", "excel"],
        "keywords": ["generate", "report", "create", "export", "how", "view"],
        "active": True
    },
    {
        "question": "How do I assign a mandate to a client?",
        "answer": "From the navbar, click Clients → select the client → navigate to the Mandate Summary page → click 'Add Mandate' → fill in the mandate details including type, duration, and permissions → click 'Save' to assign the mandate.",
        "category": "mandate_management",
        "image_path": "images/add_mandate.png",
        "related_topics": ["mandate", "client", "assign", "admin"],
        "keywords": ["assign", "mandate", "client", "add", "how", "create"],
        "active": True
    },
    {
        "question": "How do I update my profile in Visual Alpha?",
        "answer": "Click on your profile icon in the top right corner → select 'Settings' → go to 'Profile' → update your information such as name, email, and password → click 'Save' to apply the changes.",
        "category": "user_management",
        "image_path": "images/update_profile.png",
        "related_topics": ["profile", "user", "settings", "update"],
        "keywords": ["update", "profile", "settings", "user", "change", "how"],
        "active": True
    }
]


# ---------------------------
# Image Generation
# ---------------------------
def create_sample_images():
    """Create placeholder images for topics if not exist."""
    if not os.path.exists('images'):
        os.makedirs('images')
        print("📁 Created images directory")
    
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
    for cfg in image_configs:
        path = f"images/{cfg['name']}"
        if not os.path.exists(path):
            img = Image.new('RGB', (400, 200), color=cfg['color'])
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), cfg['text'], font=font)
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.text(((400-w)//2, (200-h)//2), cfg['text'], fill='white', font=font)
            img.save(path)
            created_count += 1
            print(f"✅ Created: {path}")
    if created_count == 0:
        print("ℹ️ All sample images already exist")
    else:
        print(f"🎨 Created {created_count} sample images")


# ---------------------------
# Knowledge Base
# ---------------------------
class EnhancedBusinessKnowledgeBase:
    def __init__(self, data, embedding_model):
        self.data = data
        self.embedding_model = embedding_model
        self.index = None
        self.category_index = {}
        self.keyword_index = {}
        self.build_index()

    def build_index(self):
        """Build semantic and keyword indices."""
        texts = [f"{item['question']} {item['answer']} {' '.join(item.get('related_topics', []))}" 
                 for item in self.data]
        embeddings = self.embedding_model.encode(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        for i, item in enumerate(self.data):
            cat = item.get('category', 'general')
            self.category_index.setdefault(cat, []).append(i)
            for kw in item.get('keywords', []):
                self.keyword_index.setdefault(kw.lower(), []).append(i)

    def keyword_match(self, query: str) -> List[int]:
        """Keyword-based fallback matching."""
        query_lower = query.lower().strip()
        matches = {}
        # exact phrase
        for kw in self.keyword_index:
            if kw in query_lower:
                for idx in self.keyword_index[kw]:
                    matches[idx] = matches.get(idx, 0) + len(kw.split())
        # individual words
        for w in re.findall(r'\w+', query_lower):
            if w in self.keyword_index:
                for idx in self.keyword_index[w]:
                    matches[idx] = matches.get(idx, 0) + 0.5
        return [idx for idx, score in sorted(matches.items(), key=lambda x: x[1], reverse=True) if score >= 1]

    def search(self, query: str, top_k=2, min_score=0.30) -> List[Dict]:
        """Hybrid search: semantic + keyword fallback."""
        keyword_matches = self.keyword_match(query)
        query_emb = self.embedding_model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb.astype('float32'), top_k)
        results = []

        if keyword_matches:
            idx = keyword_matches[0]
            score = 0.6 if idx not in indices[0] else float(scores[0][list(indices[0]).index(idx)])
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
        return results

    def get_related_content(self, category: str, exclude_idx=None) -> List[Dict]:
        return [self.data[i] for i in self.category_index.get(category, []) if i != exclude_idx][:2]


# ---------------------------
# Chatbot
# ---------------------------
class EnhancedBusinessChatbot:
    def __init__(self, model, tokenizer, kb):
        self.model = model
        self.tokenizer = tokenizer
        self.kb = kb
        self.conversation_history = []
        self.confidence_threshold = 0.30
        self.max_response_length = 500

    def clean_response(self, response: str) -> str:
        """Remove repeated or overly short sentences."""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        seen = set()
        cleaned = []
        for s in sentences:
            if s not in seen and len(s) > 10:
                seen.add(s)
                cleaned.append(s)
        final = '. '.join(cleaned)
        if not final.endswith('.'):
            final += '.'
        if len(final) > self.max_response_length:
            final = final[:self.max_response_length].rsplit('.', 1)[0] + '.'
        return final

    def validate_response(self, response: str, original: str) -> bool:
        words = response.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False
        if re.search(r'(.{20,}?)\1{2,}', response):
            return False
        if re.search(r'([.,;!?])\1{3,}', response):
            return False
        return True

    def generate_detailed_response(self, user_query: str) -> Tuple[str, Optional[str], List[str]]:
        user_query = user_query.strip()
        if not user_query:
            return self._handle_out_of_context("Empty query")

        results = self.kb.search(user_query, top_k=2, min_score=self.confidence_threshold)
        if not results:
            return self._handle_out_of_context(user_query)

        best = results[0]
        response = self.clean_response(best['text'])
        if not self.validate_response(response, best['text']):
            response = best['text']

        self.conversation_history.append({
            'query': user_query,
            'response': response,
            'image_path': best.get('image_path'),
            'topics': best.get('related_topics', []),
            'score': best['score'],
            'match_type': best.get('match_type', 'semantic')
        })
        return response, best.get('image_path'), best.get('related_topics', [])

    def _handle_out_of_context(self, user_query: str) -> Tuple[str, None, List[str]]:
        response = (
            "I don't have information on that. Please ask me something related to Visual Alpha. "
            "I can answer about company overview, services, team, leadership, clients, technology, history, and future goals."
        )
        return response, None, []


# ---------------------------
# Utilities
# ---------------------------
def encode_image_to_base64(image_path: str) -> Optional[str]:
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
    return None


def create_placeholder_image(text="Visual Alpha", size=(400, 200)) -> Image:
    img = Image.new('RGB', size, color='#1f2937')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((size[0]-w)//2, (size[1]-h)//2), text, fill='white', font=font)
    return img


def enhanced_chat_response(user_input: str) -> Dict:
    try:
        if chatbot is None:
            raise ValueError("Chatbot not initialized. Call initialize_bot() first.")

        response, image_path, related_topics = chatbot.generate_detailed_response(user_input)

        if image_path:
            image_data = encode_image_to_base64(image_path)
            if not image_data:
                placeholder = create_placeholder_image(f"Visual Alpha - {related_topics[0] if related_topics else 'Info'}")
                buffer = io.BytesIO()
                placeholder.save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            image_data = None

        confidence = 'high' if image_path else 'out_of_context'

        return {
            'response': response,
            'image_base64': image_data,
            'image_path': image_path,
            'related_topics': related_topics,
            'conversation_length': len(response.split()),
            'confidence': confidence
        }

    except Exception as e:
        return {
            'response': "I apologize, I encountered an error. Please try asking your question differently.",
            'image_base64': None,
            'image_path': None,
            'related_topics': [],
            'conversation_length': 0,
            'confidence': 'error'
        }


# ---------------------------
# Interactive Mode
# ---------------------------
def interactive_mode():
    print("\n🤖 Interactive Chat Mode (type 'quit' to exit)")
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not user_input:
                print("Please enter a question!")
                continue
            print("🤖 Processing... ⏳")
            result = enhanced_chat_response(user_input)
            print(f"\n🤖 Assistant ({result['conversation_length']} words):\n{result['response']}")
            if result['image_base64']:
                fname = f"chat_image_{int(time.time())}.png"
                with open(fname, 'wb') as f:
                    f.write(base64.b64decode(result['image_base64']))
                print(f"🖼️  Related image saved as: {fname}")
            if result['related_topics']:
                print(f"🏷️  Related topics: {', '.join(result['related_topics'])}")
            print(f"📊 Confidence: {result['confidence']}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    try:
        print("\n🤖 Initializing chatbot...")
        create_sample_images()
        initialize_bot()
        print("✅ Ready to chat!")
        interactive_mode()  # Start interactive mode
        # run_tests()  # Or enable tests
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
