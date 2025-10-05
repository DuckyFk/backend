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

print("🔄 Loading models... (this may take a few minutes on first run)")

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval() 
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
kb = None
chatbot = None

print("✅ Models loaded successfully!")

def initialize_bot():
    global kb, chatbot
    kb = EnhancedBusinessKnowledgeBase(business_data, embedding_model)
    chatbot = EnhancedBusinessChatbot(model, tokenizer, kb)

business_data = [
    {
        "question": "What are Visual Alpha's future goals?",
        "answer": "Visual Alpha aims to expand its client base among global firms, bringing its data-driven solutions to a broader international audience. By partnering with leading organizations worldwide, the company plans to strengthen its presence in the global financial ecosystem and continue innovating to meet the evolving needs of institutional clients.",
        "category": "future_goals",
        "image_path": "images/future_goals.png",
        "related_topics": ["global expansion", "clients", "growth", "international"]
    },

    {
        "question": "What does Visual Alpha do?",
        "answer": "Visual Alpha is a Tokyo-based B2B fintech startup offering comprehensive SaaS data solutions that revolutionize how institutional investors and asset managers handle their data operations. Our AI-powered platform automates complex data processing, generates automated reports, and provides real-time portfolio monitoring capabilities. We specialize in transforming unstructured financial data into actionable insights, reducing manual Excel work by up to 80% for investment teams. Our solutions integrate seamlessly with existing systems and provide scalable infrastructure for managing large-scale institutional portfolios.",
        "category": "company_overview",
        "image_path": "images/company_overview.png",
        "related_topics": ["services", "technology", "automation"]
    },
    {
        "question": "When was Visual Alpha founded?",
        "answer": "Visual Alpha was established in December 2019 in Tokyo, Japan, during a period of significant digital transformation in the financial sector. Since our founding, we have experienced rapid growth, expanding our client base and technology capabilities. The company was born from the vision of creating more efficient, transparent, and automated solutions for institutional investment management. Our founders leveraged their extensive experience in investment management and data systems to address the growing need for sophisticated fintech solutions in the Japanese market.",
        "category": "company_history",
        "image_path": "images/company_timeline.png",
        "related_topics": ["founding", "growth", "tokyo"]
    },
    {
        "question": "How large is the Visual Alpha team?",
        "answer": "As of 2025, Visual Alpha employs around 15-20 highly skilled professionals, including board directors, technical advisors, and core staff members. Our team represents a diverse, internationally-minded workforce with expertise spanning fintech, data science, software engineering, and financial services. We maintain a lean but highly effective organizational structure, with team members bringing experience from leading global financial institutions, technology companies, and consulting firms. Our collaborative culture emphasizes innovation, continuous learning, and delivering exceptional value to our institutional clients.",
        "category": "team_info",
        "image_path": "images/team_structure.png",
        "related_topics": ["staff", "expertise", "culture"]
    },
    {
        "question": "Who leads Visual Alpha?",
        "answer": "Visual Alpha is led by CEO Jeffrey Tsui, a seasoned professional with extensive experience in investment management data systems and financial technology. Before founding Visual Alpha, Jeffrey worked with prestigious organizations including State Street Corporation and Wellington Management, where he gained deep insights into the challenges faced by institutional investors in data management and reporting. His leadership combines technical expertise with strategic vision, driving the company's mission to transform how financial data is processed and utilized by institutional investors across Japan and beyond.",
        "category": "leadership",
        "image_path": "images/leadership_team.png",
        "related_topics": ["ceo", "experience", "background"]
    },
    {
        "question": "Who are some of Visual Alpha's clients?",
        "answer": "Visual Alpha serves a prestigious portfolio of institutional clients, including leading asset managers and pension funds in Japan. Our notable clients include Benesse Group Pension Fund, one of Japan's largest corporate pension funds; Sumitomo Mitsui DS Asset Management, a major asset management company; and Mercer Japan, a global leader in consulting services. These relationships demonstrate our ability to deliver enterprise-grade solutions that meet the stringent requirements of large-scale institutional investors, handling complex portfolio management, regulatory reporting, and data analytics needs.",
        "category": "clients",
        "image_path": "images/client_logos.png",
        "related_topics": ["institutional_investors", "pension_funds", "asset_managers"]
    },
    {
        "question": "What technologies does Visual Alpha use?",
        "answer": "Visual Alpha's technology stack is built on modern, scalable architecture designed for high-performance financial data processing. Our backend infrastructure utilizes NodeJS for server-side development, ensuring fast and efficient data processing. The frontend is built with React, providing responsive and intuitive user interfaces. We use PHP with the Laravel framework for certain web applications, while our database layer is powered by MySQL for reliable data storage. Our API architecture includes both GraphQL and RESTful APIs for flexible data access. Cloud infrastructure is managed through AWS, providing scalability and security, while Docker containers ensure consistent deployment environments. Our CI/CD pipeline is powered by CircleCI for automated testing and deployment.",
        "category": "technology",
        "image_path": "images/tech_stack.png",
        "related_topics": ["nodejs", "react", "aws", "api"]
    },
    {
        "question": "What are Visual Alpha's main services?",
        "answer": "Visual Alpha offers a comprehensive suite of financial technology services designed specifically for institutional investors. Our core services include: 1) Unstructured Data Processing - transforming complex financial documents, reports, and data feeds into structured, actionable information; 2) Content Automation - generating automated reports, presentations, and analytical documents that save hours of manual work; 3) Performance Calculation - providing accurate, real-time portfolio performance metrics and attribution analysis; 4) Portfolio Monitoring - continuous tracking of investment positions, risk metrics, and compliance requirements. All services are tailored for financial professionals managing large-scale institutional investments and integrate seamlessly with existing investment management workflows.",
        "category": "services",
        "image_path": "images/services_overview.png",
        "related_topics": ["data_processing", "automation", "portfolio_management"]
    }
]

def create_sample_images():
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
        {'name': 'services_overview.png', 'text': 'Data Processing\nAutomation', 'color': '#be185d'}
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
            print(f"✅ Created: {image_path}")
    
    if created_count == 0:
        print("ℹ️  All sample images already exist")
    else:
        print(f"🎨 Created {created_count} sample images")

class EnhancedBusinessKnowledgeBase:
    def __init__(self, data, embedding_model):
        self.data = data
        self.embedding_model = embedding_model
        self.index = None
        self.category_index = {}
        self.build_index()

    def build_index(self):
        texts = [f"{item['question']} {item['answer']} {' '.join(item.get('related_topics', []))}" 
                for item in self.data]
        embeddings = self.embedding_model.encode(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        for i, item in enumerate(self.data):
            category = item.get('category', 'general')
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(i)

    def search(self, query, top_k=2, min_score=0.25):
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if score > min_score:
                results.append({
                    'text': self.data[idx]['answer'],
                    'score': float(score),
                    'question': self.data[idx]['question'],
                    'category': self.data[idx].get('category', 'general'),
                    'image_path': self.data[idx].get('image_path'),
                    'related_topics': self.data[idx].get('related_topics', [])
                })
        return results

    def get_related_content(self, category, exclude_idx=None):
        if category in self.category_index:
            related = []
            for idx in self.category_index[category]:
                if exclude_idx is None or idx != exclude_idx:
                    related.append(self.data[idx])
            return related[:2]
        return []

class EnhancedBusinessChatbot:
    def __init__(self, model, tokenizer, knowledge_base):
        self.model = model
        self.tokenizer = tokenizer
        self.kb = knowledge_base
        self.conversation_history = []

    def generate_detailed_response(self, user_query, max_length=400) -> Tuple[str, Optional[str], List[str]]:
        relevant_docs = self.kb.search(user_query, top_k=2, min_score=0.2)
        
        if not relevant_docs:
            return self._handle_no_match(user_query)
        
        context = self._build_comprehensive_context(relevant_docs, user_query)
        prompt = self._create_prompt(context, user_query, relevant_docs)
        response = self._generate_optimized_response(prompt, max_length)
        
        image_path = relevant_docs[0].get('image_path') if relevant_docs else None
        related_topics = relevant_docs[0].get('related_topics', []) if relevant_docs else []
        
        self.conversation_history.append({
            'query': user_query,
            'response': response,
            'image_path': image_path,
            'topics': related_topics
        })
        
        return response, image_path, related_topics

    def _build_comprehensive_context(self, relevant_docs, user_query):
        context = ""
        for doc in relevant_docs:
            if doc['score'] > 0.2:
                context += f"{doc['text']}\n"
        return context.strip()

    def _create_prompt(self, context, user_query, relevant_docs):
        prompt = f"Based on: {context}\nQuestion: {user_query}\nAnswer:"
        return prompt

    def _generate_optimized_response(self, prompt, max_length):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=50,
                    num_beams=3,
                    temperature=0.6,
                    do_sample=False, 
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Generation error: {str(e)}")
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0
                    )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.strip()
            except:
                return "I apologize, but I encountered an issue generating a response. Please try asking your question in a simpler way."

    def _handle_no_match(self, user_query) -> Tuple[str, None, List[str]]:
        response = f"""I don't have specific information about "{user_query}" in my current knowledge base about Visual Alpha. 

However, I can help you with information about:
• Visual Alpha's core services and technology solutions
• Company background, leadership, and team information  
• Our client base and industry partnerships
• Technologies we use and how they benefit institutional investors
• Our founding story and growth in the Japanese fintech market

Could you please rephrase your question or ask about one of these topics? I'm designed to provide detailed information about Visual Alpha's business, services, and capabilities."""
        
        return response, None, ["services", "company_info", "technology"]

def encode_image_to_base64(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
    return None

def create_placeholder_image(text="Visual Alpha", size=(400, 200)):
    img = Image.new('RGB', size, color='#1f2937')
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

def enhanced_chat_response(user_input):
    try:
        if chatbot is None:
            raise ValueError("Chatbot not initialized. Please call initialize_bot() first.")
        response, image_path, related_topics = chatbot.generate_detailed_response(user_input)
        
        image_data = None
        if image_path:
            image_data = encode_image_to_base64(image_path)
            if not image_data:
                placeholder = create_placeholder_image(f"Visual Alpha - {related_topics[0] if related_topics else 'Info'}")
                buffer = io.BytesIO()
                placeholder.save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        result = {
            'response': response,
            'image_base64': image_data,
            'image_path': image_path,
            'related_topics': related_topics,
            'conversation_length': len(response.split()),
            'confidence': 'high' if image_path else 'medium'
        }
        
        return result
        
    except Exception as e:
        error_response = f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or ask about Visual Alpha's services, company information, or technology solutions."
        return {
            'response': error_response,
            'image_base64': None,
            'image_path': None,
            'related_topics': [],
            'conversation_length': len(error_response.split()),
            'confidence': 'error'
        }

def interactive_mode():
    print("\n🤖 Interactive Chat Mode (type 'quit' to exit)")
    print("Ask me anything about Visual Alpha!")
    print("=" * 50)
    
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
            
            print(f"\n🤖 Assistant ({result['conversation_length']} words):")
            print("-" * 40)
            print(result['response'])
            
            if result['image_base64']:
                timestamp = int(time.time())
                image_filename = f"chat_image_{timestamp}.png"
                image_data = base64.b64decode(result['image_base64'])
                with open(image_filename, 'wb') as f:
                    f.write(image_data)
                print(f"\n🖼️  Related image saved as: {image_filename}")
                print("   (Open this file to see the visual context)")
            
            if result['related_topics']:
                print(f"\n🏷️  Related topics: {', '.join(result['related_topics'])}")
            
            print(f"📊 Confidence: {result['confidence']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_tests():
    test_questions = [
        "What does Visual Alpha do?",
        "Tell me about Visual Alpha's technology stack", 
        "Who leads Visual Alpha?",
        "What are Visual Alpha's main services?",
        "Tell me about the team",
        "When was Visual Alpha founded?",
        "Who are Visual Alpha's clients?",
        "How can I contact Visual Alpha?"  
    ]
    
    print("\n🧪 Testing Enhanced Chatbot:")
    print("=" * 50)
    
    total_tests = len(test_questions)
    passed_tests = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{total_tests}] Q: {question}")
        
        try:
            start_time = time.time()
            result = enhanced_chat_response(question)
            end_time = time.time()
            
            response_time = end_time - start_time
            word_count = result['conversation_length']
            
            print(f"✅ Response ({word_count} words, {response_time:.1f}s):")
            
            response_preview = result['response'][:150] + "..." if len(result['response']) > 150 else result['response']
            print(f"   {response_preview}")
            
            print(f"🖼️  Image: {'✅ Available' if result['image_base64'] else '❌ None'}")
            print(f"🏷️  Topics: {', '.join(result['related_topics']) if result['related_topics'] else 'None'}")
            print(f"📊 Confidence: {result['confidence']}")
            
            if result['image_base64']:
                image_filename = f"test_image_{i}.png"
                image_data = base64.b64decode(result['image_base64'])
                with open(image_filename, 'wb') as f:
                    f.write(image_data)
                print(f"💾 Image saved as: {image_filename}")
            
            if (word_count >= 20 and 
                result['confidence'] != 'error' and 
                len(result['response'].strip()) > 50):
                passed_tests += 1
                print("🎯 Test PASSED")
            else:
                print("⚠️  Test issues detected")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)
    
    print(f"\n📋 Test Summary:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Chatbot is working correctly.")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Most tests passed. Minor issues may exist.")
    else:
        print("⚠️  Some tests failed. Check configuration.")

if __name__ == "__main__":
    try:
        print("\n🤖 Initializing chatbot...")
        kb = EnhancedBusinessKnowledgeBase(business_data, embedding_model)
        chatbot = EnhancedBusinessChatbot(model, tokenizer, kb)
        print("✅ Ready to chat!")
        
        interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")