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
# from langchain_community.memory import ConversationSummaryBufferMemory  # Not available in current LangChain version
# from langchain.llms import OpenAI as LangChainOpenAI  # Not available in current LangChain version

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent


class ChatbotService:
    """
    챗봇 서비스 클래스
    
    이 클래스는 챗봇의 모든 AI 로직을 캡슐화합니다.
    
    주요 책임:
    1. OpenAI API 관리
    2. ChromaDB 벡터 검색
    3. LangChain 메모리 관리
    4. 응답 생성 파이프라인
    
    직접 구현해야 할 메서드:
    - __init__: 모든 구성 요소 초기화
    - _load_config: 설정 파일 로드
    - _init_chromadb: 벡터 데이터베이스 초기화
    - _create_embedding: 텍스트 → 벡터 변환
    - _search_similar: RAG 검색 수행 (핵심!)
    - _build_prompt: 프롬프트 구성
    - generate_response: 최종 응답 생성 (모든 로직 통합)
    """
    
    def __init__(self):
        """
        챗봇 서비스 초기화
        
        TODO: 다음 구성 요소들을 초기화하세요
        
        1. Config 로드
           - config/chatbot_config.json 파일 읽기
           - 챗봇 이름, 설명, 시스템 프롬프트 등
        
        2. OpenAI Client
           - API 키: os.getenv("OPENAI_API_KEY")
           - from openai import OpenAI
           - self.client = OpenAI(api_key=...)
        
        3. ChromaDB
           - 텍스트 임베딩 컬렉션 연결
           - 경로: static/data/chatbot/chardb_embedding
           - self.collection = ...
        
        4. LangChain Memory (선택)
           - ConversationSummaryBufferMemory
           - 대화 기록 관리
           - self.memory = ...
        
        힌트:
        - ChromaDB: import chromadb
        - LangChain: # from langchain_community.memory import ConversationSummaryBufferMemory  # Not available in current LangChain version
        """
        print("[ChatbotService] 초기화 중... ")
        
        # 1. Config 로드
        self.config = self._load_config()
        
        # 2. OpenAI Client 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print("[WARNING] OPENAI_API_KEY 미설정: LLM 호출을 비활성화합니다.")
        
        # 3. ChromaDB 초기화
        self.collection = self._init_chromadb()
        
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
        
        # 5. 연애 감정 분석을 위한 키워드 로드
        self.emotion_keywords = self._load_emotion_keywords()
        
        print("[ChatbotService] 초기화 완료")
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """연애 감정 분석을 위한 키워드 로드"""
        keywords = {
            "attachment_high": ["아직도", "여전히", "지금도", "요즘도", "그리워", "보고싶어", "생각나"],
            "attachment_low": ["이제", "더 이상", "신경 안 써", "관심 없어", "잊었어", "지나간 일"],
            "regret_high": ["미안해", "아쉬워", "후회돼", "잘못했어", "다시 돌아가면", "더 잘했으면"],
            "regret_low": ["후회 없어", "그때가 최선", "맞는 선택", "다시 돌아가도"],
            "unresolved_high": ["이해가 안 돼", "궁금해", "명확하지 않아", "끝나지 않은", "해결되지 않은"],
            "unresolved_low": ["이해했어", "정리됐어", "명확해", "해결됐어", "끝났어"],
            "comparison_high": ["비교해", "그 사람만큼은", "이전과 비교하면", "새로운 사람과"],
            "comparison_low": ["비교하지 않아", "각자 다른", "독립적으로", "별개로"],
            "avoidance_high": ["피하고 싶어", "회피하고 싶어", "얘기 하기 싫어", "만나기 싫어"],
            "approach_high": ["만나고 싶어", "연락하고 싶어", "자연스럽게", "괜찮아"]
        }
        return keywords
    
    def _analyze_attachment_level(self, user_message: str) -> float:
        """애착도 분석 (0-100)"""
        high_keywords = self.emotion_keywords["attachment_high"]
        low_keywords = self.emotion_keywords["attachment_low"]
        
        high_score = sum(1 for keyword in high_keywords if keyword in user_message)
        low_score = sum(1 for keyword in low_keywords if keyword in user_message)
        
        if high_score > 0 and low_score == 0:
            return min(80 + (high_score * 5), 100)
        elif low_score > 0 and high_score == 0:
            return max(20 - (low_score * 5), 0)
        else:
            return 50  # 중립
    
    def _analyze_regret_level(self, user_message: str) -> float:
        """후회도 분석 (0-100)"""
        high_keywords = self.emotion_keywords["regret_high"]
        low_keywords = self.emotion_keywords["regret_low"]
        
        high_score = sum(1 for keyword in high_keywords if keyword in user_message)
        low_score = sum(1 for keyword in low_keywords if keyword in user_message)
        
        if high_score > 0 and low_score == 0:
            return min(80 + (high_score * 5), 100)
        elif low_score > 0 and high_score == 0:
            return max(20 - (low_score * 5), 0)
        else:
            return 50  # 중립
    
    def _analyze_unresolved_feelings(self, user_message: str) -> float:
        """미해결감 분석 (0-100)"""
        high_keywords = self.emotion_keywords["unresolved_high"]
        low_keywords = self.emotion_keywords["unresolved_low"]
        
        high_score = sum(1 for keyword in high_keywords if keyword in user_message)
        low_score = sum(1 for keyword in low_keywords if keyword in user_message)
        
        if high_score > 0 and low_score == 0:
            return min(80 + (high_score * 5), 100)
        elif low_score > 0 and high_score == 0:
            return max(20 - (low_score * 5), 0)
        else:
            return 50  # 중립
    
    def _analyze_comparison_standard(self, user_message: str) -> float:
        """비교 기준 분석 (0-100)"""
        high_keywords = self.emotion_keywords["comparison_high"]
        low_keywords = self.emotion_keywords["comparison_low"]
        
        high_score = sum(1 for keyword in high_keywords if keyword in user_message)
        low_score = sum(1 for keyword in low_keywords if keyword in user_message)
        
        if high_score > 0 and low_score == 0:
            return min(80 + (high_score * 5), 100)
        elif low_score > 0 and high_score == 0:
            return max(20 - (low_score * 5), 0)
        else:
            return 50  # 중립
    
    def _analyze_avoidance_approach(self, user_message: str) -> float:
        """회피/접근 분석 (0-100)"""
        avoidance_keywords = self.emotion_keywords["avoidance_high"]
        approach_keywords = self.emotion_keywords["approach_high"]
        
        avoidance_score = sum(1 for keyword in avoidance_keywords if keyword in user_message)
        approach_score = sum(1 for keyword in approach_keywords if keyword in user_message)
        
        if avoidance_score > approach_score:
            return min(80 + (avoidance_score * 5), 100)  # 회피
        elif approach_score > avoidance_score:
            return max(20 - (approach_score * 5), 0)  # 접근
        else:
            return 50  # 중립
    
    def _calculate_regret_index(self, user_message: str) -> Dict[str, float]:
        """종합 미련도 지수 계산"""
        attachment = self._analyze_attachment_level(user_message)
        regret = self._analyze_regret_level(user_message)
        unresolved = self._analyze_unresolved_feelings(user_message)
        comparison = self._analyze_comparison_standard(user_message)
        avoidance = self._analyze_avoidance_approach(user_message)
        
        # 가중치 적용
        total_regret = (
            attachment * 0.3 +      # 30%
            regret * 0.25 +         # 25%
            unresolved * 0.2 +      # 20%
            comparison * 0.15 +    # 15%
            avoidance * 0.1         # 10%
        )
        
        return {
            "total": total_regret,
            "attachment": attachment,
            "regret": regret,
            "unresolved": unresolved,
            "comparison": comparison,
            "avoidance": avoidance
        }
    
    def _generate_emotion_report(self, analysis_results: Dict[str, float], username: str) -> str:
        """감정 리포트 생성"""
        total = analysis_results["total"]
        
        # 미련도 지수별 해석
        if total <= 20:
            level = "완전 정리 단계"
            emoji = "💚"
            description = "이미 마음의 정리가 완전히 끝난 상태예요. 과거를 돌아보지 않고 새로운 시작을 준비하고 있어요."
        elif total <= 40:
            level = "잔잔한 여운 단계"
            emoji = "💛"
            description = "겉으로는 다 끝난 듯 보이지만, 그 시절의 따뜻함을 여전히 간직하고 있어요. '그 사람'보다는 '그때의 나'를 그리워하는 상태예요."
        elif total <= 60:
            level = "적당한 미련 단계"
            emoji = "🧡"
            description = "아직도 그 사람에 대한 감정이 남아있어요. 완전히 잊지는 못했지만, 새로운 시작을 위한 준비는 되어있어요."
        elif total <= 80:
            level = "강한 미련 단계"
            emoji = "❤️"
            description = "아직도 그 사람에 대한 강한 감정이 남아있어요. 새로운 관계를 시작하기에는 아직 시간이 더 필요할 것 같아요."
        else:
            level = "매우 강한 미련 단계"
            emoji = "💔"
            description = "아직도 그 사람에 대한 매우 강한 감정이 남아있어요. 완전한 정리가 필요해 보여요."
        
        # 주요 감정 키워드 추출
        keywords = []
        if analysis_results["attachment"] > 60:
            keywords.append("#그리움")
        if analysis_results["regret"] > 60:
            keywords.append("#후회")
        if analysis_results["unresolved"] > 60:
            keywords.append("#미해결감")
        if analysis_results["comparison"] > 60:
            keywords.append("#비교")
        if analysis_results["avoidance"] > 60:
            keywords.append("#회피")
        
        if not keywords:
            keywords = ["#성장", "#이해", "#정리"]
        
        report = f"""[{username}님의 연애 감정 리포트]

1️⃣ 주요 감정 키워드
{' '.join(keywords)}

2️⃣ 감정 상태 분석
"{description}"

3️⃣ 미련도 지수
{emoji} **{int(total)}% — {level}**

4️⃣ 개인화된 메시지
"""
        
        # 개인화된 조언 추가
        if total <= 20:
            report += "과거를 아름답게 정리하고 새로운 시작을 준비하고 있는 모습이 정말 멋져요. 이제 진짜 새로운 사랑을 만날 준비가 되어있어요!"
        elif total <= 40:
            report += "아직도 그 시절의 따뜻함을 간직하고 있지만, 이제는 '그 사람'보다는 '그때의 나'를 그리워하고 있어요. 이는 정말 건강한 감정이에요!"
        elif total <= 60:
            report += "아직도 그 사람에 대한 감정이 남아있지만, 이제는 새로운 시작을 위한 준비가 되어있어요. 조금 더 시간을 갖고 천천히 나아가세요!"
        elif total <= 80:
            report += "아직도 그 사람에 대한 강한 감정이 남아있어요. 새로운 관계를 시작하기에는 아직 시간이 더 필요할 것 같아요. 조금 더 기다려보세요!"
        else:
            report += "아직도 그 사람에 대한 매우 강한 감정이 남아있어요. 완전한 정리가 필요해 보여요. 전문가의 도움을 받는 것도 좋은 방법이에요!"
        
        return report
    
    
    def _load_config(self):
        """
        설정 파일 로드
        
        TODO: config/chatbot_config.json 읽어서 반환
        
        반환값 예시:
        {
            "name": "김서강",
            "character": {...},
            "system_prompt": {...}
        }
        """
        config_path = BASE_DIR / "config" / "chatbot_config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] 설정 파일을 찾을 수 없습니다: {config_path}")
            return {
                "name": "환승연애 PD 친구",
                "description": "환승연애팀 막내 PD 친구",
                "system_prompt": {
                    "base": "당신은 환승연애팀 막내 PD가 된 친구입니다.",
                    "rules": ["친근하게 대화하세요", "연애 이야기를 자연스럽게 이끌어내세요"]
                }
            }
    
    
    def _init_chromadb(self):
        """
        ChromaDB 초기화 및 컬렉션 반환
        
        TODO: 
        1. PersistentClient 생성
        2. 컬렉션 가져오기 (이름: "rag_collection")
        3. 컬렉션 반환
        
        힌트:
        - import chromadb
        - db_path = BASE_DIR / "static/data/chatbot/chardb_embedding"
        - client = chromadb.PersistentClient(path=str(db_path))
        - collection = client.get_collection(name="rag_collection")
        """
        db_path = BASE_DIR / "static/data/chatbot/chardb_embedding"
        db_path.mkdir(parents=True, exist_ok=True)
        
        client = None
        try:
            client = chromadb.PersistentClient(path=str(db_path))
            try:
                collection = client.get_collection(name="rag_collection")
                print(f"[ChromaDB] 컬렉션 연결 성공: {collection.name}")
                return collection
            except Exception:
                # 없으면 생성
                collection = client.create_collection(name="rag_collection")
                print(f"[ChromaDB] 새 컬렉션 생성: {collection.name}")
                return collection
        except Exception as e:
            print(f"[WARNING] ChromaDB 초기화 실패: {e}")
            return None
    
    
    def _create_embedding(self, text: str) -> list:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str): 임베딩할 텍스트
        
        Returns:
            list: 3072차원 벡터 (text-embedding-3-large 모델)
        
        TODO:
        1. OpenAI API 호출
        2. embeddings.create() 사용
        3. 벡터 반환
        
        힌트:
        - response = self.client.embeddings.create(
        -     input=[text],
        -     model="text-embedding-3-large"
        - )
        - return response.data[0].embedding
        """
        if not self.client:
            return []
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[ERROR] 임베딩 생성 실패: {e}")
            return []
    
    
    def _search_similar(self, query: str, threshold: float = 0.45, top_k: int = 5):
        """
        RAG 검색: 유사한 문서 찾기 (핵심 메서드!)
        
        Args:
            query (str): 검색 질의
            threshold (float): 유사도 임계값 (0.3-0.5 권장)
            top_k (int): 검색할 문서 개수
        
        Returns:
            tuple: (document, similarity, metadata) 또는 (None, None, None)
        
        TODO: RAG 검색 알고리즘 구현
        
        1. 쿼리 임베딩 생성
           query_embedding = self._create_embedding(query)
        
        2. ChromaDB 검색
           results = self.collection.query(
               query_embeddings=[query_embedding],
               n_results=top_k,
               include=["documents", "distances", "metadatas"]
           )
        
        3. 유사도 계산 및 필터링
           for doc, dist, meta in zip(...):
               similarity = 1 / (1 + dist)  ← 유사도 공식!
               if similarity >= threshold:
                   ...
        
        4. 가장 유사한 문서 반환
           return (best_document, best_similarity, metadata)
        
        
        💡 핵심 개념:
        
        - Distance vs Similarity
          · ChromaDB는 "거리(distance)"를 반환 (작을수록 유사)
          · 우리는 "유사도(similarity)"로 변환 (클수록 유사)
          · 변환 공식: similarity = 1 / (1 + distance)
        
        - Threshold
          · 0.3: 매우 느슨한 매칭 (관련성 낮아도 OK)
          · 0.45: 적당한 매칭 (추천!)
          · 0.7: 매우 엄격한 매칭 (정확한 답만)
        
        - Top K
          · 5-10개 정도 검색
          · 그 중 threshold 넘는 것만 사용
        
        
        🐛 디버깅 팁:
        - print()로 검색 결과 확인
        - 유사도 값 확인 (너무 낮으면 threshold 조정)
        - 검색된 문서 내용 확인
        """
        if not self.collection:
            print("[WARNING] ChromaDB 컬렉션이 없습니다.")
            return None, None, None
        
        try:
            # 1. 쿼리 임베딩 생성 (LLM 비활성화 시 RAG 생략)
            if not self.client:
                return None, None, None
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return None, None, None
            
            # 2. ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            
            # 3. 유사도 계산 및 필터링
            best_document = None
            best_similarity = 0
            best_metadata = None
            
            if results['documents'] and results['documents'][0]:
                for doc, dist, meta in zip(
                    results['documents'][0], 
                    results['distances'][0], 
                    results['metadatas'][0]
                ):
                    similarity = 1 / (1 + dist)  # 유사도 공식
                    print(f"[RAG] 유사도: {similarity:.4f}, 거리: {dist:.4f}")
                    
                    if similarity >= threshold and similarity > best_similarity:
                        best_document = doc
                        best_similarity = similarity
                        best_metadata = meta
            
            if best_document:
                print(f"[RAG] 최고 유사도: {best_similarity:.4f}")
                print(f"[RAG] 문서: {best_document[:100]}...")
                return best_document, best_similarity, best_metadata
            else:
                print(f"[RAG] 임계값({threshold}) 이상의 유사한 문서를 찾지 못했습니다.")
                return None, None, None
                
        except Exception as e:
            print(f"[ERROR] RAG 검색 실패: {e}")
            return None, None, None
    
    
    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자"):
        """
        LLM 프롬프트 구성
        
        Args:
            user_message (str): 사용자 메시지
            context (str): RAG 검색 결과 (선택)
            username (str): 사용자 이름
        
        Returns:
            str: 최종 프롬프트
        
        TODO:
        1. 시스템 프롬프트 가져오기 (config에서)
        2. RAG 컨텍스트 포함 여부 결정
        3. 대화 기록 포함 (선택)
        4. 최종 프롬프트 문자열 반환
        
        프롬프트 예시:
        ```
        당신은 서강대학교 선배 김서강입니다.
        신입생들에게 학교 생활을 알려주는 역할을 합니다.
        
        [참고 정보]  ← RAG 컨텍스트가 있을 때만
        학식은 곤자가가 맛있어. 돈까스가 인기야.
        
        사용자: 학식 추천해줘
        ```
        """
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
        
        # 사용자 메시지 추가
        prompt_parts.append(f"\n{username}: {user_message}")
        
        return "\n".join(prompt_parts)
    
    
    def generate_response(self, user_message: str, username: str = "사용자") -> dict:
        """
        사용자 메시지에 대한 챗봇 응답 생성
        
        Args:
            user_message (str): 사용자 입력
            username (str): 사용자 이름
        
        Returns:
            dict: {
                'reply': str,       # 챗봇 응답 텍스트
                'image': str|None   # 이미지 경로 (선택)
            }
        
        
        TODO: 전체 응답 생성 파이프라인 구현
        
        
        ═══════════════════════════════════════════════════
        📋 구현 단계
        ═══════════════════════════════════════════════════
        
        [1단계] 초기 메시지 처리
        
            if user_message.strip().lower() == "init":
                # 첫 인사말 반환
                bot_name = self.config.get('name', '챗봇')
                return {
                    'reply': f"안녕! 나는 {bot_name}이야.",
                    'image': None
                }
        
        
        [2단계] RAG 검색 수행
        
            context, similarity, metadata = self._search_similar(
                query=user_message,
                threshold=0.45,
                top_k=5
            )
            
            has_context = (context is not None)
        
        
        [3단계] 프롬프트 구성
        
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username
            )
        
        
        [4단계] LLM API 호출
        
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 또는 gpt-4
                messages=[
                    {"role": "system", "content": "시스템 프롬프트"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            reply = response.choices[0].message.content
        
        
        [5단계] 메모리 저장 (선택)
        
            if self.memory:
                self.memory.save_context(
                    {"input": user_message},
                    {"output": reply}
                )
        
        
        [6단계] 응답 반환
        
            return {
                'reply': reply,
                'image': None  # 이미지 검색 로직 추가 가능
            }
        
        
        ═══════════════════════════════════════════════════
        💡 핵심 포인트
        ═══════════════════════════════════════════════════
        
        1. RAG 활용
           - 검색 결과가 있으면 프롬프트에 포함
           - 없으면 일반 대화 모드
        
        2. 에러 처리
           - try-except로 API 오류 처리
           - 실패 시 기본 응답 반환
        
        3. 로깅
           - 각 단계마다 print()로 상태 출력
           - 디버깅에 매우 유용!
        
        4. 확장성
           - 이미지 검색 로직 추가 가능
           - 감정 분석 추가 가능
           - 다중 언어 지원 가능
        
        
        ═══════════════════════════════════════════════════
        🐛 디버깅 예시
        ═══════════════════════════════════════════════════
        
        print(f"\n{'='*50}")
        print(f"[USER] {username}: {user_message}")
        print(f"[RAG] Context found: {has_context}")
        if has_context:
            print(f"[RAG] Similarity: {similarity:.4f}")
            print(f"[RAG] Context: {context[:100]}...")
        print(f"[LLM] Calling API...")
        print(f"[BOT] {reply}")
        print(f"{'='*50}\n")
        """
        
        # 여기에 전체 파이프라인 구현
        # 위의 단계를 참고하여 자유롭게 설계하세요
        
        try:
            print(f"\n{'='*50}")
            print(f"[USER] {username}: {user_message}")
            
            # [1단계] 초기 메시지 처리
            if user_message.strip().lower() == "init":
                bot_name = self.config.get('name', '환승연애 PD 친구')
                return {
                    'reply': f"야, {username}! 나 이번에 환승연애 팀 막내 PD 됐잖아. 근데 지금 새 프로그램 기획 중인데, 솔직히 사람들 연애 얘기 좀 모으고 있어. 너 전 연애 얘기 좀 해줄 수 있어?",
                    'image': None
                }
            
            # [2단계] RAG 검색 수행
            context, similarity, metadata = self._search_similar(
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
            analysis_results = self._calculate_regret_index(user_message)
            print(f"[ANALYSIS] 미련도: {analysis_results['total']:.1f}%")
            
            # [4단계] 프롬프트 구성
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username
            )
            
            # [5단계] LLM API 호출
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
                    report = self._generate_emotion_report(analysis_results, username)
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
