"""
🎯 챗봇 서비스 - 구현 파일

이 파일은 챗봇의 핵심 AI 로직을 담당합니다.
아래 아키텍처를 참고하여 직접 설계하고 구현하세요.

📐 시스템 아키텍처:

┌─────────────────────────────────────────────────────────┐
│ 1. 초기화 단계 (ChatbotService.__init__)                  │
├─────────────────────────────────────────────────────────┤
│  - OpenAI Client 생성                                    │
│  - ChromaDB 연결 (벡터 데이터베이스)                       │
│  - LangChain Memory 초기화 (대화 기록 관리)               │
│  - Config 파일 로드                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. RAG 파이프라인 (generate_response 내부)               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  사용자 질문 "학식 추천해줘"                              │
│       ↓                                                  │
│  [_create_embedding()]                                   │
│       ↓                                                  │
│  질문 벡터: [0.12, -0.34, ..., 0.78]  (3072차원)        │
│       ↓                                                  │
│  [_search_similar()]  ← ChromaDB 검색                    │
│       ↓                                                  │
│  검색 결과: "학식은 곤자가가 맛있어" (유사도: 0.87)        │
│       ↓                                                  │
│  [_build_prompt()]                                       │
│       ↓                                                  │
│  최종 프롬프트 = 시스템 설정 + RAG 컨텍스트 + 질문        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. LLM 응답 생성                                         │
├─────────────────────────────────────────────────────────┤
│  OpenAI GPT-4 API 호출                                   │
│       ↓                                                  │
│  "학식은 곤자가에서 먹는 게 제일 좋아! 돈까스가 인기야"    │
│       ↓                                                  │
│  [선택: 이미지 검색]                                      │
│       ↓                                                  │
│  응답 반환: {reply: "...", image: "..."}                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 메모리 저장 (LangChain Memory)                        │
├─────────────────────────────────────────────────────────┤
│  대화 기록에 질문-응답 저장                               │
│  다음 대화에서 컨텍스트로 활용                            │
└─────────────────────────────────────────────────────────┘


💡 핵심 구현 과제:

1. **Embedding 생성**
   - OpenAI API를 사용하여 텍스트를 벡터로 변환
   - 모델: text-embedding-3-large (3072차원)

2. **RAG 검색 알고리즘** ⭐ 가장 중요!
   - ChromaDB에서 유사 벡터 검색
   - 유사도 계산: similarity = 1 / (1 + distance)
   - threshold 이상인 문서만 선택

3. **LLM 프롬프트 설계**
   - 시스템 프롬프트 (캐릭터 설정)
   - RAG 컨텍스트 통합
   - 대화 기록 포함

4. **대화 메모리 관리**
   - LangChain의 ConversationSummaryBufferMemory 사용
   - 대화가 길어지면 자동으로 요약


📚 참고 문서:
- ARCHITECTURE.md: 시스템 아키텍처 상세 설명
- IMPLEMENTATION_GUIDE.md: 단계별 구현 가이드
- README.md: 프로젝트 개요


⚠️ 주의사항:
- 이 파일의 구조는 가이드일 뿐입니다
- 자유롭게 재설계하고 확장할 수 있습니다
- 단, generate_response() 함수 시그니처는 유지해야 합니다
  (app.py에서 호출하기 때문)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Tuple, Optional
import chromadb
from openai import OpenAI
from .emotion_analyzer import EmotionAnalyzer, ReportGenerator
from .rag_service import RAGService
from .config_loader import ConfigLoader
# from langchain_community.memory import ConversationSummaryBufferMemory  # Not available in current LangChain version
# from langchain.llms import OpenAI as LangChainOpenAI  # Not available in current LangChain version

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent


class ChatbotService:

    
    def __init__(self):
 
        print("[ChatbotService] 초기화 중... ")
        
        # 1. Config 로드
        self.config = ConfigLoader.load_config()
        
        # 2. OpenAI Client 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print("[WARNING] OPENAI_API_KEY 미설정: LLM 호출을 비활성화합니다.")
        
        # 3. RAG 서비스 초기화
        self.rag_service = RAGService(self.client)
        
        # 4. LangChain Memory 초기화 (API 키가 있을 때만)
        self.memory = None
        if api_key:
            try:
                llm = LangChainOpenAI(openai_api_key=api_key, temperature=0.7)
                self.memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=1000,
                    return_messages=True
                )
            except Exception as e:
                print(f"[WARNING] 메모리 초기화 실패: {e}")
        
        # 5. 감정 분석 서비스 초기화
        self.emotion_analyzer = EmotionAnalyzer()
        self.report_generator = ReportGenerator()
        
        # 6. DSM 상태 관리 변수 초기화
        self.dialogue_state = 'INTRO'  # 대화 상태 (INTRO, RECALL_ATTACHMENT, RECALL_REGRET, etc.)
        self.turn_count = 0  # 대화 턴 수 추적
        self.stop_request_count = 0  # 사용자 대화 중단 요청 횟수
        
        print("[ChatbotService] 초기화 완료")
    
    
    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자"):
 
        # 시스템 프롬프트 구성
        system_prompt = self.config.get('system_prompt', {})
        base_prompt = system_prompt.get('base', '당신은 환승연애팀 막내 PD가 된 친구입니다.')
        rules = system_prompt.get('rules', [])
        
        # 기본 프롬프트 구성
        prompt_parts = [base_prompt]
        
        # 규칙 추가
        if rules:
            prompt_parts.append("\n".join([f"- {rule}" for rule in rules]))
        
        # RAG 컨텍스트 추가
        if context:
            prompt_parts.append(f"\n[참고 정보]\n{context}")
        
        # 대화 기록 추가 (선택)
        if self.memory:
            try:
                memory_vars = self.memory.load_memory_variables({})
                if memory_vars and 'history' in memory_vars:
                    prompt_parts.append(f"\n[대화 기록]\n{memory_vars['history']}")
            except Exception as e:
                print(f"[WARNING] 메모리 로드 실패: {e}")
        
        # 대화 지침 추가
        prompt_parts.append("\n대화 지침:")
        prompt_parts.append("- 친구처럼 편하게 반말로 대화해")
        prompt_parts.append("- 너무 상세하게 계속 물어보지 말고, 적당한 타이밍에 다른 주제로 넘어가")
        prompt_parts.append("- 연애 이야기를 자연스럽게 이끌어내되, 무리하게 끌어내지 마")
        prompt_parts.append("- 이모티콘은 최소한으로 사용해")
        
        # Redirection Rule 및 주제 이탈 방지 지침 추가
        prompt_parts.append("\n[PD 친구 규칙 강화]:")
        prompt_parts.append("- 너는 환승연애 PD 친구로서, 오직 전애인(X)과의 연애 이야기에만 집중해야 해.")
        prompt_parts.append("\n[주제 복귀 규칙]:")
        prompt_parts.append("- 사용자가 현애인 또는 전애인과 무관한 주제(일반 일상, 미래 계획 등)로 대화가 이탈하면, 'AI 분석 범위 밖' 또는 '기획안 데이터'를 핑계로 친근하게 대화를 전애인 이야기로 복귀시켜야 해. 절대로 딱딱하게 끊거나 강압적으로 들리면 안 돼.")
        
        # 사용자 메시지 추가
        prompt_parts.append(f"\n{username}: {user_message}")
        
        return "\n".join(prompt_parts)
    
    
    def generate_response(self, user_message: str, username: str = "사용자") -> dict:
        
        
        # 여기에 전체 파이프라인 구현
        # 위의 단계를 참고하여 자유롭게 설계하세요
        
        try:
            print(f"\n{'='*50}")
            print(f"[USER] {username}: {user_message}")
            
            # [1단계] 초기 메시지 처리
            if user_message.strip().lower() == "init":
                bot_name = self.config.get('name', '환승연애 PD 친구')
                # 도입부: INTRO 상태로 시작
                self.dialogue_state = 'INTRO'
                self.turn_count = 0
                self.stop_request_count = 0
                return {
                    'reply': f"야, {username}! 요즘 나 일 재밌어 죽겠어ㅋㅋ 나 드디어 환승연애 막내 PD 됐다니까! 근데 웃긴 게, 요즘 거기서 AI 도입 얘기가 진짜 많아. 다음 시즌엔 무려 ‘X와의 미련도 측정 AI’ 같은 것도 넣는대ㅋㅋㅋ 완전 신박하지 않아? 내가 요즘 그거 관련해서 연애 사례 모으고 있거든. 가만 생각해보니까… 너 얘기가 딱이야. 아직 테스트 버전이라 진짜 재미삼아 보는 거야. 부담 갖지마마 그냥 친구한테 옛날 얘기하듯이 편하게 말해줘 ㅋㅋ 너 예전에 그 X 있잖아. 혹시 X랑 있었던 일 얘기해줄 수 있어?",

                    'image': None
                }
            
            # 일반 메시지의 경우 turn_count 증가
            self.turn_count += 1
            
            # [2단계] RAG 검색 수행
            #우리는 RAG 검색 매 질문마다 사용 하지 않음 불필요 
            context, similarity, metadata = self.rag_service.search_similar(
                query=user_message,
                threshold=0.45,
                top_k=5
            )
            
            has_context = (context is not None)
            print(f"[RAG] Context found: {has_context}")
            if has_context:
                print(f"[RAG] Similarity: {similarity:.4f}")
                print(f"[RAG] Context: {context[:100]}...")
            
            # [3단계] 연애 감정 분석 수행
            analysis_results = self.emotion_analyzer.calculate_regret_index(user_message)
            print(f"[ANALYSIS] 미련도: {analysis_results['total']:.1f}%")
            
            # [4단계] 프롬프트 구성
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username
            )
            
            # [5단계] LLM API 호출
            # 불필요한 중복인가?
            if self.client:
                print(f"[LLM] Calling API...")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 환승연애팀 막내 PD가 된 친구입니다. 사용자와 반말로 자연스럽게 대화하며, 연애 이야기를 듣고 미련도를 분석해주는 역할을 합니다. 친구처럼 편하게 대화하고, 이모티콘은 최소한으로 사용하세요. 너무 상세하게 계속 물어보지 말고, 적당한 타이밍에 다른 주제로 넘어가거나 분석 결과를 제시하세요. 자연스러운 대화 흐름을 유지하세요."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
            else:
                # LLM 비활성화 시 기본 응답
                reply = "AI 연애 분석 에이전트 데모 모드야. 환경변수 설정 후 더 정교한 분석이 가능해! 먼저 어떤 이야기부터 시작할까?"
            
            # [6단계] 감정 리포트 생성 (특정 조건에서)
            if any(keyword in user_message.lower() for keyword in ["분석", "리포트", "결과", "어때", "어떤"]):
                if analysis_results['total'] > 0:  # 분석 결과가 있을 때만
                    report = self.report_generator.generate_emotion_report(analysis_results, username)
                    reply += f"\n\n{report}"
            
            # [7단계] 메모리 저장
            if self.memory:
                try:
                    self.memory.save_context(
                        {"input": user_message},
                        {"output": reply}
                    )
                except Exception as e:
                    print(f"[WARNING] 메모리 저장 실패: {e}")
            
            print(f"[BOT] {reply[:100]}...")
            print(f"{'='*50}\n")
            
            # [8단계] 응답 반환
            return {
                'reply': reply,
                'image': None
            }
            
        except Exception as e:
            print(f"[ERROR] 응답 생성 실패: {e}")
            return {
                'reply': "죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요.",
                'image': None
            }


# ============================================================================
# 싱글톤 패턴
# ============================================================================
# ChatbotService 인스턴스를 앱 전체에서 재사용
# (매번 새로 초기화하면 비효율적)

_chatbot_service = None

def get_chatbot_service():
    """
    챗봇 서비스 인스턴스 반환 (싱글톤)
    
    첫 호출 시 인스턴스 생성, 이후 재사용
    """
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


# ============================================================================
# 테스트용 메인 함수
# ============================================================================

if __name__ == "__main__":
    """
    로컬 테스트용
    
    실행 방법:
    python services/chatbot_service.py
    """
    print("챗봇 서비스 테스트")
    print("=" * 50)
    
    service = get_chatbot_service()
    
    # 초기화 테스트
    response = service.generate_response("init", "테스터")
    print(f"초기 응답: {response}")
    
    # 일반 대화 테스트
    response = service.generate_response("안녕하세요!", "테스터")
    print(f"응답: {response}")
