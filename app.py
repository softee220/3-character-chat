"""
🚫 이 파일은 수정하지 마세요! (템플릿 파일)

이 파일은 Flask 애플리케이션의 핵심 로직을 포함하고 있습니다.
학회원은 다음 파일만 수정/작성하면 됩니다:

✏️ 수정/작성해야 하는 파일:
  - config/chatbot_config.json        (챗봇 설정)
  - services/chatbot_service.py       (AI 로직: RAG, Embedding, LLM)
  - static/data/chatbot/chardb_text/  (텍스트 데이터)
  - static/images/chatbot/            (이미지 파일)
  - static/videos/chatbot/            (비디오 파일, 선택)

이 파일을 수정하면 전체 시스템이 작동하지 않을 수 있습니다.
"""

import os
import json
from pathlib import Path
from flask import Flask, request, render_template, jsonify, url_for
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-this')

# 개발 환경 설정
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent

# 설정 파일 로드
CONFIG_PATH = BASE_DIR / 'config' / 'chatbot_config.json'

def load_config():
    """챗봇 설정 파일 로드"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본 설정 반환
        return {
            'name': '챗봇',
            'description': '챗봇 설명',
            'tags': ['#챗봇'],
            'thumbnail': 'images/hateslop/club_logo.png'
        }

config = load_config()

# 이미지 파일 스캔 함수
def get_image_files():
    """챗봇 이미지 디렉토리에서 이미지 파일 목록 반환"""
    folder_path = BASE_DIR / "static" / "images" / "chatbot"
    image_files = []
    
    if folder_path.exists():
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                    image_files.append(rel_path.replace("\\", "/"))
    
    return image_files

# 메인 페이지
@app.route('/')
def index():
    bot_info = {
        'name': config.get('name', '챗봇'),
        'image': url_for('static', filename=config.get('thumbnail', 'images/hateslop/club_logo.png')),
        'tags': config.get('tags', ['#챗봇']),
        'description': config.get('description', '')
    }
    return render_template('index.html', bot=bot_info)

# 챗봇 상세정보 페이지
@app.route('/detail')
def detail():
    bot_info = {
        'name': config.get('name', '챗봇'),
        'image': url_for('static', filename=config.get('thumbnail', 'images/hateslop/club_logo.png')),
        'description': config.get('description', ''),
        'tags': config.get('tags', ['#챗봇'])
    }
    return render_template('detail.html', bot=bot_info)

# 채팅 화면
@app.route('/chat')
def chat():
    username = request.args.get('username', '사용자')
    bot_name = config.get('name', '챗봇')
    image_files = get_image_files()
    
    return render_template('chat.html', 
                         bot_name=bot_name, 
                         username=username,
                         image_files=image_files)

# API 엔드포인트: 챗봇 응답 생성
@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        username = data.get('username', '사용자')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # 챗봇 서비스 임포트 (지연 로딩)
        from services import get_chatbot_service
        
        # 응답 생성
        chatbot = get_chatbot_service()
        response = chatbot.generate_response(user_message, username)
        
        return jsonify(response)
        
    except ImportError as e:
        print(f"[ERROR] 챗봇 서비스 임포트 실패: {e}")
        return jsonify({'reply': '챗봇 서비스를 불러올 수 없습니다. services/chatbot_service.py를 구현해주세요.'}), 500
    except Exception as e:
        print(f"[ERROR] 응답 생성 실패: {e}")
        return jsonify({'reply': '죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요.'}), 500

# 프로그램 추천 페이지
@app.route('/program_recommendation')
def program_recommendation():
    """프로그램 추천 페이지"""
    image = request.args.get('image', '')
    message = request.args.get('message', '')
    sentiment = request.args.get('sentiment', 'positive')
    
    return render_template('program_recommendation.html', 
                         image=image,
                         message=message,
                         sentiment=sentiment)

# 헬스체크 엔드포인트 (Vercel용)
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'chatbot': config.get('name', 'unknown')})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
