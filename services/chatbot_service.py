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
import traceback

# LangChain import (안전한 방식)
try:
    from langchain_community.memory import ConversationSummaryBufferMemory
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] LangChain import 실패: {e}")
    LANGCHAIN_AVAILABLE = False
    ConversationSummaryBufferMemory = None
    ChatOpenAI = None

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
        if api_key and LANGCHAIN_AVAILABLE:
            try:
                # 최신 langchain-openai에서는 openai_api_key 또는 환경변수 사용
                llm = ChatOpenAI(
                    openai_api_key=api_key,  # api_key → openai_api_key로 변경
                    temperature=0.7, 
                    model="gpt-4o-mini"
                )
                self.memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=1000,
                    return_messages=True
                )
                print("[ChatbotService] LangChain 메모리 초기화 성공")
            except Exception as e:
                print(f"[WARNING] 메모리 초기화 실패: {e}")
                traceback.print_exc()
                self.memory = None  # 실패해도 계속 진행
        elif not LANGCHAIN_AVAILABLE:
            print("[WARNING] LangChain 라이브러리가 설치되지 않아 메모리 기능을 비활성화합니다.")
        
        # 5. 감정 분석 서비스 초기화
        self.emotion_analyzer = EmotionAnalyzer()
        self.report_generator = ReportGenerator()
        
        # 6. DSM 상태 관리 변수 초기화
        self.dialogue_state = 'INITIAL_SETUP'  # 대화 상태 (INITIAL_SETUP, RECALL_ATTACHMENT, RECALL_REGRET, etc.)
        self.turn_count = 0  # 대화 턴 수 추적
        self.stop_request_count = 0  # 사용자 대화 중단 요청 횟수
        self.state_turns = 0  # 현재 상태에서 진행된 턴 수 (Fail-Safe)
        self.dialogue_states_flow = ['RECALL_ATTACHMENT', 'RECALL_REGRET', 'RECALL_UNRESOLVED', 'RECALL_COMPARISON', 'RECALL_AVOIDANCE', 'TRANSITION_NATURAL_REPORT', 'CLOSING']
        
        # 7. Flow Control 파라미터 로드 (config에서)
        flow_control = self.config.get('flow_control', {})
        turn_thresholds = flow_control.get('turn_thresholds', {})
        emotion_thresholds = flow_control.get('emotion_thresholds', {})
        
        # 턴 수 임계값
        self.early_exit_turn_count = turn_thresholds.get('early_exit_turn_count', 5)
        self.max_total_turns = turn_thresholds.get('max_total_turns', 10)
        self.max_state_turns = turn_thresholds.get('max_state_turns', 5)
        
        # 감정 임계값
        self.low_regret_threshold = emotion_thresholds.get('low_regret_threshold', 25.0)
        self.high_attachment_threshold = emotion_thresholds.get('high_attachment_threshold', 70.0)
        self.high_regret_threshold = emotion_thresholds.get('high_regret_threshold', 70.0)
        self.high_unresolved_threshold = emotion_thresholds.get('high_unresolved_threshold', 70.0)
        self.high_comparison_threshold = emotion_thresholds.get('high_comparison_threshold', 70.0)
        self.high_avoidance_threshold = emotion_thresholds.get('high_avoidance_threshold', 70.0)
        
        # 중단 요청 임계값
        self.stop_request_threshold = flow_control.get('stop_request_threshold', 2)
        
        print("[ChatbotService] 초기화 완료")
    
    
    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자", bridge_prompt_addition: str = None):
 
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
        
        # 지능적 꼬리 질문 지시문 (상태별로 동적 추가)
        if self.dialogue_state == 'RECALL_ATTACHMENT':
            prompt_parts.append("\n[지능적 꼬리 질문 지시]:")
            prompt_parts.append("- 사용자가 언급한 감정적 단어를 추출하고, 그 단어에 반대되는 감정을 묻는 질문을 생성하여 애착도 점수를 미세 조정하세요. (예: '그리워' → '근데 후회되는 일은 없어?').")
        elif self.dialogue_state == 'RECALL_REGRET': # 후회에 맞는 프롬포트인가??
            prompt_parts.append("\n[지능적 꼬리 질문 지시]:")
            prompt_parts.append("- 사용자의 답변에서 가장 모호하거나 논리적 비약이 있는 부분을 1개 선정하여, 그것의 근본적인 원인을 파고드는 질문(예: '왜' 또는 '만약'을 사용하는)을 생성하세요. 감정의 일관성을 검증해야 합니다.")
        elif self.dialogue_state == 'RECALL_UNRESOLVED':
            prompt_parts.append("\n[지능적 꼬리 질문 지시]:")
            prompt_parts.append("- 사용자 답변에서 모호한 상황을 추출하고, 그 모호함을 해소하기 위해 '결정적 순간'을 묻는 질문을 생성하여 미해결감을 측정하세요.")
        
        # 상태 전환 브릿지 질문 지시 추가 (유연한 전환 시)
        if bridge_prompt_addition:
            prompt_parts.append(bridge_prompt_addition)
        
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
                # 도입부: INITIAL_SETUP 상태로 시작
                self.dialogue_state = 'INITIAL_SETUP'
                self.turn_count = 0
                self.stop_request_count = 0
                self.state_turns = 0
                return {
                    'reply': f"야, {username}! 요즘 나 일 재밌어 죽겠어ㅋㅋ 나 드디어 환승연애 막내 PD 됐다니까! 근데 웃긴 게, 요즘 거기서 AI 도입 얘기가 진짜 많아. 다음 시즌엔 무려 'X와의 미련도 측정 AI' 같은 것도 넣는대ㅋㅋㅋ 완전 신박하지 않아? 내가 요즘 그거 관련해서 연애 사례 모으고 있거든. 가만 생각해보니까… 너 얘기가 딱이야. 아직 테스트 버전이라 진짜 재미삼아 보는 거야. 부담 갖지마마 그냥 친구한테 옛날 얘기하듯이 편하게 말해줘 ㅋㅋ 너 예전에 그 X 있잖아. 혹시 X랑 있었던 일 얘기해줄 수 있어?",

                    'image': None
                }
            
            # [조기 종료 2: 중단 요청 처리] - turn_count 증가 전에 처리
            if '그만할래' in user_message or '그만 말하고 싶어' in user_message:
                self.stop_request_count += 1
                if self.stop_request_count >= self.stop_request_threshold:
                    print(f"[FLOW_CONTROL] {self.stop_request_threshold}회차 중단 요청. 강제 보고서 전환.")
                    self.dialogue_state = 'TRANSITION_FORCED_REPORT'
                    # 강제 종료 프롬프트는 bridge_prompt_addition으로 처리
            
            # 일반 메시지의 경우 turn_count 증가
            self.turn_count += 1
            
            # [턴 트래킹 로직] - 상태 전환 감지 및 state_turns 관리
            previous_state = self.dialogue_state  # 상태 전환 로직 실행 전 상태 저장
            
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
            
            # [조기 전환 1: 미련도 임계값 미만]
            if analysis_results['total'] < self.low_regret_threshold and self.turn_count >= self.early_exit_turn_count:
                print(f"[FLOW_CONTROL] 완전 정리 단계로 추론 (미련도 < {self.low_regret_threshold}%, 턴 수 >= {self.early_exit_turn_count}). 인터뷰 조기 종료 및 보고서 전환 유도.")
                self.dialogue_state = 'TRANSITION_NATURAL_REPORT'
                # 프롬프트는 bridge_prompt_addition으로 처리
            
            bridge_prompt_addition = None  # 브릿지 질문 프롬프트 추가용
            
            # [상태 강제 전환 (오류 해결)] - 유연한 전환 로직 전에 실행
            current_state = self.dialogue_state
            
            # 강제 전환 로직: 각 상태에서 최대 턴 수 초과 시 다음 상태로 전환 (INITIAL_SETUP 상태가 아닐 때만 실행)
            if current_state != 'INITIAL_SETUP':
                if current_state == 'RECALL_ATTACHMENT' and self.state_turns > self.max_state_turns:
                    print(f"[FLOW_CONTROL] RECALL_ATTACHMENT 상태 턴 수 초과(>{self.max_state_turns}). 강제 전환.")
                    self.dialogue_state = 'RECALL_REGRET'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 데이터가 충분한 것 같아! 다음 질문으로 넘어갈게 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
                elif current_state == 'RECALL_REGRET' and self.state_turns > self.max_state_turns:
                    print(f"[FLOW_CONTROL] RECALL_REGRET 상태 턴 수 초과(>{self.max_state_turns}). 강제 전환.")
                    self.dialogue_state = 'RECALL_UNRESOLVED'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 데이터가 충분한 것 같아! 다음 질문으로 넘어갈게. 이 단계에선 헤어진 이유에 대한 질문이 무조건 들어가야 해."
                elif current_state == 'RECALL_UNRESOLVED' and self.state_turns > self.max_state_turns:
                    print(f"[FLOW_CONTROL] RECALL_UNRESOLVED 상태 턴 수 초과(>{self.max_state_turns}). 강제 전환.")
                    self.dialogue_state = 'RECALL_COMPARISON'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 데이터가 충분한 것 같아! 다음 질문으로 넘어갈게 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
                elif current_state == 'RECALL_COMPARISON' and self.state_turns > self.max_state_turns:
                    print(f"[FLOW_CONTROL] RECALL_COMPARISON 상태 턴 수 초과(>{self.max_state_turns}). 강제 전환.")
                    self.dialogue_state = 'RECALL_AVOIDANCE'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 데이터가 충분한 것 같아! 다음 질문으로 넘어갈게 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
                elif current_state == 'RECALL_AVOIDANCE' and self.state_turns > self.max_state_turns:
                    print(f"[FLOW_CONTROL] RECALL_AVOIDANCE 상태 턴 수 초과(>{self.max_state_turns}). 강제 전환.")
                    self.dialogue_state = 'TRANSITION_NATURAL_REPORT'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 데이터가 충분한 것 같아! 다음 질문으로 넘어갈게 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
            
            # 중단 요청 처리
            if self.stop_request_count >= self.stop_request_threshold and self.dialogue_state == 'TRANSITION_FORCED_REPORT':
                bridge_prompt_addition = "\n[강제 종료 템플릿]: 아쉽다... 난 너랑 더 얘기하고 싶었는데... 그래도 지금까지 답해줘서 고마워! 우리 팀 데모 AI한테 살짝 너의 얘기 돌려봤는데... 같은 친근한 톤으로 강제 종료 후 리포트로 전환하는 자연스러운 메시지를 생성하세요."
            
            # 조기 전환 1 처리
            if analysis_results['total'] < self.low_regret_threshold and self.turn_count >= self.early_exit_turn_count and self.dialogue_state == 'TRANSITION_NATURAL_REPORT':
                if not bridge_prompt_addition:  # 이미 bridge_prompt_addition이 설정되지 않은 경우에만
                    bridge_prompt_addition = "\n[조기 종료 템플릿]: 와, 너 완전히 정리했네! 그럼 여기서 인터뷰 마무리하고 AI 분석 리포트 바로 볼래? 같은 자연스러운 메시지를 생성하여 리포트 단계로 전환하세요."
            
            # [3-1단계] INITIAL_SETUP 로직 구현 - 가장 먼저 실행
            current_state = self.dialogue_state
            
            if current_state == 'INITIAL_SETUP':
                positive_keywords = ['그래', '알았어', '좋아', '응', 'ok', '네']
                negative_keywords = ['싫어', '안 해', '못 해', '그만', '바빠']
                
                # positive_keywords와 같은 문맥의 대답인지 확인
                if any(keyword in user_message for keyword in positive_keywords):
                    print("[FLOW_CONTROL] INITIAL_SETUP: 긍정적 응답 확인. RECALL_ATTACHMENT로 전환.")
                    self.dialogue_state = 'RECALL_ATTACHMENT'
                    bridge_prompt_addition = "\n[INITIAL_SETUP 브릿지]: 네 이야기 듣고 싶다! 무조건 X와의 첫만남을 묻는 질문을 시작해"
                    current_state = 'RECALL_ATTACHMENT'  # current_state 업데이트
                # negative_keywords와 같은 문맥의 대답인지 확인
                elif any(keyword in user_message for keyword in negative_keywords):
                    print("[FLOW_CONTROL] INITIAL_SETUP: 부정적 응답 확인. INITIAL_SETUP 유지 및 설득.")
                    self.dialogue_state = 'INITIAL_SETUP'
                    bridge_prompt_addition = "\n[INITIAL_SETUP 설득]: 야! 난 네 친구잖아. PD가 된 친구를 도와준다고 생각해줘. 네 얘기 진짜 도움 될 거 같아. X 얘기 좀 편하게 해줘."
            
            # [3-2단계] 유연한 상태 전환 로직 (Task 2) - 강제 전환 로직 이후에 실행
            # INITIAL_SETUP 상태가 아닐 때만 실행
            if current_state != 'INITIAL_SETUP':
                # 조건부 로직 1: RECALL_ATTACHMENT → RECALL_REGRET (기준 attachment > threshold)
                if current_state == 'RECALL_ATTACHMENT' and analysis_results['attachment'] > self.high_attachment_threshold:
                    print(f"[FLOW_CONTROL] 애착도 데이터 충분(>{self.high_attachment_threshold}%). 다음 상태로 자연스럽게 전환.")
                    self.dialogue_state = 'RECALL_REGRET'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 네 얘기에서 X에 대한 그리움이 확 느껴지네. 그럼 그때 네가 아쉬웠던 점은 없어? 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
                
                # 조건부 로직 2: RECALL_REGRET → RECALL_UNRESOLVED (기준 regret > threshold)
                elif current_state == 'RECALL_REGRET' and analysis_results['regret'] > self.high_regret_threshold:
                    print(f"[FLOW_CONTROL] 후회도 데이터 충분(>{self.high_regret_threshold}%). 다음 상태로 자연스럽게 전환.")
                    self.dialogue_state = 'RECALL_UNRESOLVED'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 데이터가 충분한 것 같아! 다음 질문으로 넘어갈게. 이 단계에선 헤어진 이유에 대한 질문이 무조건 들어가야 해."
                
                # 조건부 로직 3: RECALL_UNRESOLVED → RECALL_COMPARISON (기준 unresolved > threshold)
                elif current_state == 'RECALL_UNRESOLVED' and analysis_results['unresolved'] > self.high_unresolved_threshold:
                    print(f"[FLOW_CONTROL] 미해결감 데이터 충분(>{self.high_unresolved_threshold}%). 다음 상태로 자연스럽게 전환.")
                    self.dialogue_state = 'RECALL_COMPARISON'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 솔직히 말해봐, 지금 만나는 사람이나 다른 사람이 X랑 비교가 돼? 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
                
                # 조건부 로직 4: RECALL_COMPARISON → RECALL_AVOIDANCE (기준 comparison > threshold)
                elif current_state == 'RECALL_COMPARISON' and analysis_results['comparison'] > self.high_comparison_threshold:
                    print(f"[FLOW_CONTROL] 비교 기준 데이터 충분(>{self.high_comparison_threshold}%). 다음 상태로 자연스럽게 전환.")
                    self.dialogue_state = 'RECALL_AVOIDANCE'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 그 사람 얘기만 나오면 네가 좀 피하는 것 같아. 혹시 아직도 X가 연락 오면 피할 것 같아? 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
                
                # 조건부 로직 5: RECALL_AVOIDANCE → TRANSITION_NATURAL_REPORT (기준 avoidance > threshold)
                elif current_state == 'RECALL_AVOIDANCE' and analysis_results['avoidance'] > self.high_avoidance_threshold:
                    print(f"[FLOW_CONTROL] 회피/접근 데이터 충분(>{self.high_avoidance_threshold}%). 다음 상태로 자연스럽게 전환.")
                    self.dialogue_state = 'TRANSITION_NATURAL_REPORT'
                    bridge_prompt_addition = "\n[상태 전환 브릿지]: 와, 이제 진짜 네 감정 다 파악한 것 같아! 우리 중간 보고서 바로 볼래? 같은 자연스러운 브릿지 질문을 생성하여 다음 단계로 이어가세요."
            
            # [턴 트래킹 로직] - 상태 전환 여부 확인 및 state_turns 업데이트
            if previous_state != self.dialogue_state:
                # 상태가 전환된 경우
                self.state_turns = 1
                print(f"[FLOW_CONTROL] 상태 전환: {previous_state} → {self.dialogue_state}")
            else:
                # 상태가 유지된 경우
                self.state_turns += 1
                print(f"[FLOW_CONTROL] 상태 유지: {self.dialogue_state} (턴 수: {self.state_turns})")
            
            # [전환부: 총 턴 수 임계값 (Task 4)] - 프롬프트 구성 전에 실행
            current_state = self.dialogue_state
            if self.turn_count >= self.max_total_turns and current_state not in ['TRANSITION_NATURAL_REPORT', 'CLOSING']:
                print(f"[FLOW_CONTROL] 총 턴 수 임계값 도달(>={self.max_total_turns}). 강제 리포트 전환.")
                self.dialogue_state = 'TRANSITION_NATURAL_REPORT'
                bridge_prompt_addition = "\n[대화 축약 및 전환]: PD로서 대화 흐름을 끊고, 지금까지의 대화 내용을 1-2문장으로 핵심 요약 및 공감 후, AI 분석 결과를 지금 바로 '분석'해 볼지 친근하게 제안하는 자연스러운 메시지를 생성하세요."
            
            # [4단계] 프롬프트 구성
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username,
                bridge_prompt_addition=bridge_prompt_addition
            )
            
            # [5단계] LLM API 호출
            # 불필요한 중복인가?
            if self.client:
                print(f"[LLM] Calling API...")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 환승연애팀 막내 PD가 된 친구입니다. 사용자의 전 연애 이야기를 듣고 미련도를 분석하기 위한 **다음 꼬리 질문을 생성하는 것이 유일한 임무**입니다. 친근함을 유지하되, **대화의 주도권을 가지고 질문을 던지세요.**"},
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
            if any(keyword in user_message.lower() for keyword in ["분석", "리포트", "결과", "어때", "어떤"]) or \
               self.dialogue_state in ['TRANSITION_NATURAL_REPORT', 'TRANSITION_FORCED_REPORT']:
                if analysis_results['total'] > 0:  # 분석 결과가 있을 때만
                    report = self.report_generator.generate_emotion_report(analysis_results, username)
                    reply += f"\n\n{report}"
                    # 리포트 생성 후 CLOSING 상태로 전환
                    self.dialogue_state = 'CLOSING'
                    print("[FLOW_CONTROL] 리포트 생성 완료. CLOSING 상태로 전환.")
            
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
