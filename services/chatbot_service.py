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
from .response_generator import ResponseGenerator
import traceback

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
        
        # 4. 대화 기록 저장소 초기화
        self.dialogue_history: List[Dict[str, str]] = []
        
        # 5. 감정 분석 서비스 초기화 (RAG, OpenAI 클라이언트 주입)
        self.emotion_analyzer = EmotionAnalyzer(rag_service=self.rag_service, openai_client=self.client)
        self.report_generator = ReportGenerator(rag_service=self.rag_service, openai_client=self.client)
        
        # 5. DSM 상태 관리 변수 초기화
        self.dialogue_state = 'INITIAL_SETUP'  # 대화 상태
        self.turn_count = 0  # 대화 턴 수 추적
        self.stop_request_count = 0  # 사용자 대화 중단 요청 횟수
        self.state_turns = 0  # 현재 상태에서 진행된 턴 수
        self.dialogue_states_flow = ['RECALL_UNRESOLVED', 'RECALL_ATTACHMENT', 'RECALL_REGRET', 'RECALL_COMPARISON', 'RECALL_AVOIDANCE', 'TRANSITION_NATURAL_REPORT', 'CLOSING']
        self.final_regret_score = None  # 리포트 생성 시점의 최종 미련도 점수 저장
        
        # 6. 고정 질문 시스템 초기화
        self.fixed_questions = self.config.get('fixed_questions', {})
        self.question_indices = {}  # 각 상태별 현재 질문 인덱스
        self.tail_question_used = {}  # 각 상태별 꼬리 질문 사용 여부
        
        # 초기화: 모든 상태의 질문 인덱스를 0으로 설정
        for state in self.fixed_questions.keys():
            self.question_indices[state] = 0
            self.tail_question_used[state] = False
        
        # 7. Flow Control 파라미터 로드 (config에서)
        flow_control = self.config.get('flow_control', {})
        turn_thresholds = flow_control.get('turn_thresholds', {})
        emotion_thresholds = flow_control.get('emotion_thresholds', {})
        
        # 턴 수 임계값
        self.early_exit_turn_count = turn_thresholds.get('early_exit_turn_count', 5)
        self.max_total_turns = turn_thresholds.get('max_total_turns', 25) 
        # 하드 코딩해야할듯....
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
        
        # 8. 이미지 매핑 설정
        self.image_mapping = {
            'empathy': 'images/chatbot/01_empathy.png',  # 공감
            'unconditional_support': 'images/chatbot/01_support.png',  # 무조건적인 지지
            'surprise': 'images/chatbot/01_surprised.png',  # 놀람
            'firm_advice': 'images/chatbot/01_advice.png',  # 단호한 조언
            'laughing': 'images/chatbot/01_smile.png',  # 웃는 모습
            'careful': 'images/chatbot/01_careful.png'  # 눈치보는 모습
        }
        
        # 9. 응답 생성기 초기화
        self.response_generator = ResponseGenerator(self)
        
        print("[ChatbotService] 초기화 완료")
    
    
    def _detect_report_feedback(self, user_message: str) -> bool:
        """
        리포트에 대한 피드백인지 감지합니다.
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            피드백이면 True, 그렇지 않으면 False
        """
        feedback_keywords = [
            '어때', '어떤', '어떻게 생각', '생각해', '생각이', '생각해?', '생각해요',
            '맞아', '맞다고', '그래', '그렇구나', '알겠어', '이해했어',
            '재밌어', '좋아', '괜찮아', '괜찮네', '재미있어',
            '신기해', '대박', '와', '헐', '진짜', '와우',
            '그렇네', '그런가', '흠', '음', '아', '오',
            '결과', '리포트', '분석', '점수', '미련도',
            '어울려', '어울리', '프로그램', '프로그램이'
        ]
        
        message_lower = user_message.lower()
        
        # 리포트 피드백 키워드 포함 여부 확인
        return any(keyword in message_lower for keyword in feedback_keywords)
    
    def _select_image_by_response(self, reply: str) -> Optional[str]:
        """
        AI 응답 내용을 분석하여 적절한 이미지를 선택합니다.
        
        Args:
            reply: AI가 생성한 응답 텍스트
            
        Returns:
            이미지 경로 (/static/... 형태) 또는 None
        """
        reply_lower = reply.lower()
        
        # 키워드 기반 이미지 선택 로직
        # 우선순위: 단호한 조언 > 지지 > 눈치 > 공감 > 놀람 > 웃는 모습
        
        selected_image = None
        
        # 1. 단호한 조언 (최우선) - 가장 명확한 감정 표현
        if any(keyword in reply_lower for keyword in ['해야 해', '해야겠어', '해야 할', '해야 돼', '필요해', '중요해', '무조건', '절대', '반드시', 
                               '제발', '꼭', '해봐', '하세요', '하자', '조언', '추천', '해줘', '해봐봐']):
            selected_image = self.image_mapping['firm_advice']
        
        # 2. 무조건적인 지지 - 응원과 격려 표현
        elif any(keyword in reply_lower for keyword in ['응원', '힘내', '화이팅', '넌 할 수 있어', '믿어', '멋져', '잘했어', 
                          '고생했어', '수고했어', '훌륭해', '대단해', '다 괜찮아질 거야', '좋아', '좋네', '좋다']):
            selected_image = self.image_mapping['unconditional_support']
        
        # 3. 눈치보는 모습 - 조심스러운 표현
        elif any(keyword in reply_lower for keyword in ['혹시', '괜찮아?', '불편하면', '부담', '아니면', '안 되면', '싫으면', 
                          '원치 않으면', '괜찮으면', '괜찮다면', '괜찮아?', '괜찮아']):
            selected_image = self.image_mapping['careful']
        
        # 4. 공감 - 공감과 이해 표현 (키워드 확장)
        elif any(keyword in reply_lower for keyword in ['알겠어', '이해해', '같아', '맞아', '그렇구나', '공감', '느껴', '알 것 같아', 
                          '이해', '알겠다', '그런가', '그런 것 같아', '동감', '맞다고', '그래', '그렇지', '그렇군', '그렇구나',
                          '아하', '아 그렇구나', '아 그렇군', '그런 거', '그런 거네', '그런 것 같아', '느낌', '느껴져']):
            selected_image = self.image_mapping['empathy']
        
        # 5. 놀람 - 명확한 놀람 표현
        elif any(keyword in reply_lower for keyword in ['와', '헐', '대박', '와우', '오마이갓', '놀랐어', '놀랐다', '놀라', '신기해', '신기하다']):
            selected_image = self.image_mapping['surprise']
        
        # 6. 웃는 모습 (가장 마지막 우선순위)
        elif any(keyword in reply_lower for keyword in ['ㅋㅋㅋ','ㅎㅎㅎ', '웃겨', '웃기', '재밌어', '재밌네', '재밌다', '웃었어', '웃었네', '웃었지', '웃음', '웃고', '유쾌']):
            selected_image = self.image_mapping['laughing']
        
        # 기본값: 공감 (가장 일반적인 반응)
        if selected_image is None:
            selected_image = self.image_mapping['empathy']
        
        # Flask static 경로로 변환
        if selected_image:
            return f"/static/{selected_image}"
        
        return None
    
    
    def _get_next_question(self, state: str) -> Optional[str]:
        """
        현재 상태의 다음 고정 질문을 가져옵니다.
        
        Args:
            state: DSM 상태
            
        Returns:
            다음 고정 질문 문자열, 없으면 None
        """
        if state not in self.fixed_questions:
            return None
        
        questions = self.fixed_questions[state]
        current_idx = self.question_indices.get(state, 0)
        
        if current_idx < len(questions):
            return questions[current_idx]
        return None
    
    
    def _is_questions_exhausted(self, state: str) -> bool:
        """
        현재 상태의 고정 질문을 모두 소진했는지 확인합니다.
        
        Args:
            state: DSM 상태
            
        Returns:
            True if 모든 질문 소진, False otherwise
        """
        if state not in self.fixed_questions:
            return True
        
        questions = self.fixed_questions[state]
        current_idx = self.question_indices.get(state, 0)
        
        return current_idx >= len(questions)
    
    
    def _mark_question_used(self, state: str):
        """
        현재 질문을 사용 완료로 표시하고 인덱스를 증가시킵니다.
        """
        if state not in self.question_indices:
            self.question_indices[state] = 0
        self.question_indices[state] += 1
        print(f"[QUESTION] {state} 상태: 질문 인덱스 → {self.question_indices[state]}")
    
    
    def get_next_state_in_flow(self, current_state: str) -> Tuple[Optional[str], Optional[str]]:
        """
        현재 상태의 다음 상태와 그 상태의 첫 번째 고정 질문(오프너)을 반환합니다.
        
        Args:
            current_state: 현재 DSM 상태
            
        Returns:
            (다음 상태, 다음 상태의 첫 번째 고정 질문) 튜플. 없으면 (None, None)
        """
        try:
            current_idx = self.dialogue_states_flow.index(current_state)
            if current_idx + 1 < len(self.dialogue_states_flow):
                next_state = self.dialogue_states_flow[current_idx + 1]
                # 다음 상태의 첫 번째 고정 질문을 오프너로 가져옴
                if next_state in self.fixed_questions and len(self.fixed_questions[next_state]) > 0:
                    next_opener = self.fixed_questions[next_state][0]
                    return (next_state, next_opener)
                return (next_state, None)
        except ValueError:
            pass
        return (None, None)
    
    
    def _detect_topic_deviation(self, user_message: str) -> Optional[str]:
        """
        사용자 메시지에서 주제 이탈을 감지합니다.
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            redirect 타입 ("current_future_relationship" or "personal_topic") 또는 None
        """
        current_future_keywords = ['현애인', '지금 만나는', '다음 연애', '미래', '새로운 사람', '현재', '지금']
        personal_keywords = ['일상', '취미', '가족', '학교', '회사', '여행']
        
        message_lower = user_message.lower()
        
        # 현애인/미래 주제 이탈
        if any(keyword in message_lower for keyword in current_future_keywords):
            return "current_future_relationship"
        
        # 사적 주제 이탈 (간단한 휴리스틱, 필요시 확장)
        personal_count = sum(1 for keyword in personal_keywords if keyword in message_lower)
        if personal_count >= 2:  # 사적 키워드가 2개 이상 포함되면
            return "personal_topic"
        
        return None
    
    
    def _detect_no_ex_story(self, user_message: str) -> bool:
        """
        X 스토리 부재를 감지합니다 (문맥 기반).
        
        주의: 부정적 답변("싫어", "안 해")과 구분해야 합니다.
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            X 스토리가 없으면 True, 그렇지 않으면 False
        """
        # X 부재 키워드
        no_ex_keywords = [
            '없는데', '없어', '없다', '없음',
            '안 해봤', '못 해봤', '해본 적',
            '모솔', '솔로', '연애 경험'
        ]
        
        # 부정 답변 키워드 (이건 제외)
        refusal_keywords = ['싫어', '안 해', '그만', '바빠']
        
        message_lower = user_message.lower()
        
        # 부정 답변이면 False (기존 중단 요청 로직으로 처리)
        if any(kw in message_lower for kw in refusal_keywords):
            return False
        
        # X 부재 키워드 1개 이상 감지
        return any(kw in message_lower for kw in no_ex_keywords)
    
    
    def _is_short_answer(self, user_message: str) -> bool:
        """
        사용자 답변이 짧은지 감지합니다.
        
        Args:
            user_message: 사용자 메시지
            
        Returns:
            답변이 짧으면 True, 그렇지 않으면 False
        """
        # 공백 제거 후 길이 체크
        message_trimmed = user_message.strip()
        
        # 20자 이하면 짧은 답변으로 간주
        if len(message_trimmed) <= 15:
            return True
        
        # 짧은 답변 키워드 체크
        short_answer_keywords = [
            '그래', '그렇지', '그렇네', '그렇구나', '그래요', '그렇다',
            '몰라', '모르겠어', '모르겠다', '모르겠네',
            '그냥', '그저', '그런데', '그럼',
            '응', '어', '음', '으음', '아', '아하',
            '맞아', '맞아요', '맞다', '맞네',
            '있어', '없어', '있었어', '없었어',
            '좋아', '좋아요', '싫어', '싫어요',
            '알겠어', '알겠다', '알겠네'
        ]
        
        message_lower = message_trimmed.lower()
        
        # 짧은 답변 키워드가 포함되어 있고, 전체 메시지가 짧으면
        if any(kw in message_lower for kw in short_answer_keywords) and len(message_trimmed) <= 20:
            return True
        
        return False
    
    
    def _is_question_already_asked(self, question: str, lookback_turns: int = 10) -> bool:
        """
        최근 대화 기록에서 같은 질문이 이미 했는지 확인합니다.
        
        Args:
            question: 확인할 질문
            lookback_turns: 확인할 대화 턴 수 (기본 10턴)
            
        Returns:
            이미 했던 질문이면 True, 그렇지 않으면 False
        """
        if not self.dialogue_history:
            return False
        
        question_lower = question.lower().strip()
        
        # 최근 대화 기록에서 봇 메시지만 확인
        recent_bot_messages = []
        for item in reversed(self.dialogue_history[-lookback_turns*2:]):
            if item.get('role') == '혜슬' or item.get('role') == '이다음':
                recent_bot_messages.append(item.get('content', '').lower())
        
        # 유사한 질문이 있는지 확인 (의미 유사도 체크)
        for bot_msg in recent_bot_messages:
            # 질문의 핵심 키워드 추출하여 비교
            question_keywords = set([q.strip() for q in question_lower.split() if len(q.strip()) > 2])
            bot_msg_keywords = set([b.strip() for b in bot_msg.split() if len(b.strip()) > 2])
            
            # 70% 이상 키워드가 겹치면 중복으로 판단
            if question_keywords and bot_msg_keywords:
                overlap = len(question_keywords & bot_msg_keywords) / len(question_keywords)
                if overlap > 0.7:
                    return True
        
        return False
    
    
    def _create_empathy_first_instruction(self, question: str, context: str = "") -> str:
        """
        공감 우선 구조 프롬프트를 생성하는 공통 메서드.
        
        Args:
            question: 던질 질문
            context: 추가 맥락 (선택사항)
            
        Returns:
            공감 우선 구조 프롬프트 문자열
        """
        base = f"[공감 우선]: 공감(2-3문장) → 자연스러운 전환어 → 질문: {question}"
        return f"{context}\n{base}" if context else base
    
    
    def _generate_bridge_question_prompt(self, current_state: str, next_state: str, transition_reason: str) -> str:
        """
        상태 전환 시 브릿지 질문 생성을 위한 프롬프트를 생성합니다.
        
        Args:
            current_state: 현재 상태
            next_state: 다음 상태
            transition_reason: 전환 이유
            
        Returns:
            브릿지 프롬프트 문자열
        """
        next_question = self._get_next_question(next_state)
        # 중복 질문 체크: 이미 한 질문이면 다음 질문으로 넘어감
        if next_question and self._is_question_already_asked(next_question):
            self._mark_question_used(next_state)
            next_question = self._get_next_question(next_state)
        
        # 사용자의 마지막 답변 추출 (대화 기록에서)
        last_user_message = ""
        if len(self.dialogue_history) >= 2:
            # 마지막 메시지가 봇이면, 그 전이 사용자 메시지
            for item in reversed(self.dialogue_history[-4:]):
                if item.get('role') != '혜슬' and item.get('role') != '이다음':
                    last_user_message = item.get('content', '')[:100]  # 최근 100자만
                    break
        
        # 공감 우선 구조 사용 (공통 메서드 활용)
        context = f"[상태 전환] {current_state} → {next_state} ({transition_reason})\n사용자 마지막 답변: \"{last_user_message}\""
        if current_state == 'RECALL_UNRESOLVED' and next_state == 'RECALL_ATTACHMENT':
            context += "\n이별 맥락 후 긍정적 기억으로 자연스럽게 전환."
        
        bridge_prompt = self._create_empathy_first_instruction(next_question, context)
        return bridge_prompt
    
    
    def _generate_closing_proposal_prompt(self, recent_dialogue: List[Dict[str, str]]) -> str:
        """
        대화 종료 제안 프롬프트를 생성합니다.
        
        Args:
            recent_dialogue: 최근 대화 기록
            
        Returns:
            종료 제안 프롬프트 문자열
        """
        closing_prompt = """
[대화 종료 제안]

네 이야기를 들어보니 [대화 내용 1~2문장 핵심 요약 및 공감] 같은데,
더 깊은 이야기는 나중에 더 해보자.
내가 아까 말한 우리 팀 데모 AI 에이전트에 네 데이터 충분히 들어간 것 같거든?
재미삼아 AI 분석 결과를 지금 바로 **'분석'**해 볼래?
분석을 원하면 말해줘!
"""
        return closing_prompt
    
    def _collect_dialogue_context_for_report(self) -> str:
        """
        리포트 생성을 위한 대화 맥락 수집
        
        Returns:
            str: 사용자의 주요 답변들을 묶은 텍스트
        """
        user_responses = []
        for item in self.dialogue_history:
            # 혜슬(봇)의 메시지가 아닌 것만 수집
            if item.get('role') != '혜슬':
                user_responses.append(item.get('content', ''))
        
        # 최근 10개 사용자 답변만 사용 (너무 길어지지 않도록)
        context = "\n\n".join(user_responses[-10:])
        return context
    
    
    def _build_prompt(self, user_message: str, username: str = "사용자", special_instruction: str = None):
        """
        현재 턴의 지시사항과 사용자 메시지를 구성합니다.
        
        Args:
            user_message: 사용자 메시지
            username: 사용자 이름
            special_instruction: 특별 지시사항 (브릿지, redirect 등)
        """
        prompt_parts = []
        
        # 최근 대화 요약 (반복 방지) - 더 긴 범위로 확장
        if len(self.dialogue_history) >= 10:
            recent_turns = self.dialogue_history[-10:]
            recent_summary = "\n".join([f"{item['role']}: {item['content'][:80]}..." for item in recent_turns])
            prompt_parts.append(f"""[최근 대화 요약 - CRITICAL: 이미 물어본 질문은 절대 반복하지 마]
{recent_summary}

**중요: 위 대화 요약을 꼭 확인하고, 이미 물어본 질문이나 사용자가 이미 답변한 내용에 대해 다시 물어보지 마.**
""")
        elif len(self.dialogue_history) >= 6:
            recent_turns = self.dialogue_history[-6:]
            recent_summary = "\n".join([f"{item['role']}: {item['content'][:80]}..." for item in recent_turns])
            prompt_parts.append(f"""[최근 대화 요약 - CRITICAL: 이미 물어본 질문은 절대 반복하지 마]
{recent_summary}

**중요: 위 대화 요약을 꼭 확인하고, 이미 물어본 질문이나 사용자가 이미 답변한 내용에 대해 다시 물어보지 마.**
""")
        
        # 짧은 답변 감지
        is_short = self._is_short_answer(user_message)
        is_uncertain_answer = any(keyword in user_message.lower() for keyword in ['몰라', '모르겠어', '모르겠다', '모르겠네', '기억 안 나', '기억이 안 나'])
        
        # 이미 답변했다는 신호 감지 ("아까 말했잖아", "이미 답했어" 등)
        already_answered_keywords = ['아까', '이미', '말했', '답했', '했잖아', '했어', '알았잖아', '알았어', '말한', '했던']
        is_already_answered = any(keyword in user_message.lower() for keyword in already_answered_keywords) and len(user_message) < 30
        
        # 짧은 답변 처리는 하이브리드 전략에서 자연스럽게 처리됨 (화제 전환 신호로 사용하지 않음)
        
        # 이미 답변했다는 신호 처리
        if is_already_answered:
            # 다음 참고 질문으로 넘어가거나 다음 주제로 전환
            next_state, next_opener = self.get_next_state_in_flow(self.dialogue_state)
            current_questions = self.fixed_questions.get(self.dialogue_state, [])
            current_q_idx = self.question_indices.get(self.dialogue_state, 0)
            remaining_questions = current_questions[current_q_idx:] if current_q_idx < len(current_questions) else []
            
            if remaining_questions and len(remaining_questions) > 0:
                # 같은 주제의 다음 참고 질문으로 넘어가기 (인덱스 증가)
                next_question_hint = remaining_questions[0]
                # 질문 인덱스 증가 (같은 질문 반복 방지)
                self._mark_question_used(self.dialogue_state)
                prompt_parts.append(f"""[이미 답변함 감지]
사용자가 "아까 말했잖아" 같은 답변을 했어. 이미 물어본 질문을 반복하지 말고, 아래 다른 각도 질문으로 넘어가거나 사용자의 답변에서 새로운 궁금한 점을 찾아봐.

[다음 참고 질문 (방향성만 참고)]
{next_question_hint}

**중요: 위 질문을 그대로 말하지 말고, 같은 맥락에서 다른 각도로 자연스럽게 질문해.**
""")
            elif next_state and next_opener:
                # 다음 주제로 전환
                prompt_parts.append(f"""[이미 답변함 감지 - 화제 전환]
사용자가 이미 답변했다고 했어. 현재 주제에 대해 충분히 정보를 얻었으니, 아래 '다음 주제 오프너'를 사용해서 자연스럽게 화제를 전환해.

[다음 주제 오프너]
"{next_opener}"
""")
            else:
                prompt_parts.append("[이미 답변함 감지]: 사용자가 이미 답변했다고 했어. 같은 질문을 반복하지 말고, 사용자의 답변에서 새로운 궁금한 점이나 다른 각도를 찾아봐.")
        
        # 불확실한 답변 처리
        elif is_uncertain_answer:
            prompt_parts.append("[공감 우선]: '몰라' 같은 답변은 힘들었거나 기억하고 싶지 않았을 수 있음. 먼저 진심으로 공감(2-3문장) 후 자연스럽게 대화 이어가거나 다른 각도로 접근. 질문 강제 금지.")
        
            # 하이브리드 전략: state_turns 기반 지시 (이미 답변함 감지 시 제외)
        elif (not is_already_answered and 
              self.dialogue_state in self.fixed_questions and 
              self.dialogue_state not in ['TRANSITION_NATURAL_REPORT', 'TRANSITION_FORCED_REPORT', 'CLOSING', 'NO_EX_CLOSING', 'REPORT_SHOWN', 'FINAL_CLOSING']):
            next_state, next_opener = self.get_next_state_in_flow(self.dialogue_state)
            
            # 현재 상태의 목표와 참고 질문들 가져오기
            current_questions = self.fixed_questions.get(self.dialogue_state, [])
            current_q_idx = self.question_indices.get(self.dialogue_state, 0)
            remaining_questions = current_questions[current_q_idx:] if current_q_idx < len(current_questions) else []
            
            # 상태별 목표 설명
            state_goals = {
                'RECALL_UNRESOLVED': '헤어진 이유와 미해결된 감정(unresolved)',
                'RECALL_ATTACHMENT': 'X에 대한 애착과 연상(attachment)',
                'RECALL_REGRET': '후회와 아쉬움(regret)',
                'RECALL_COMPARISON': 'X와의 비교와 대조(comparison)',
                'RECALL_AVOIDANCE': '회피와 거리두기(avoidance)'
            }
            current_goal = state_goals.get(self.dialogue_state, '현재 주제')
            
            # 특별 지시사항이 있으면 하이브리드 전략을 건너뛰고 special_instruction만 사용
            if special_instruction:
                # special_instruction이 있으면 여기서는 아무것도 추가하지 않음 (special_instruction이 나중에 추가됨)
                pass
            # 상태 전환 직후 첫 턴(state_turns == 0)일 때는 참고 질문 보여주지 않음
            elif self.state_turns == 0:
                # 첫 턴이면 기본 지시만 (하나의 질문만)
                prompt_parts.append(f"[대화 지침]\n사용자의 마지막 답변에 공감하고, 현재 주제({current_goal})에 대해 자연스럽게 질문해. **오직 하나의 질문만** 해.")
            elif self.state_turns < 2:
                # 초반부: 꼬리 질문에만 집중하되, 상태 목표 명시
                if remaining_questions:
                    prompt_parts.append(f"""[대화 지침]
사용자의 마지막 답변에 공감하고, 그 답변 내용 중에서 더 궁금한 점이나 깊게 파고들 부분을 자연스럽게 질문해. **오직 하나의 질문만** 해.

[현재 상태 목표]
지금은 '{current_goal}'에 대한 정보를 얻는 중이야. 아래 참고 질문들을 보면서, 같은 맥락에서 자연스러운 꼬리 질문을 해.

[참고 질문들 - 방향성만 참고하고 절대 그대로 말하지 마]
{chr(10).join([f"- {q}" for q in remaining_questions[:3]])}
""")
                else:
                    prompt_parts.append(f"[대화 지침]\n사용자의 마지막 답변에 공감하고, 그 답변 내용 중에서 더 궁금한 점이나 깊게 파고들 부분을 자연스럽게 질문해. **오직 하나의 질문만** 해. 현재 주제({current_goal})에 대해 더 깊이 들어가.")
            else:
                # 중반부 이후: 선택권 부여
                if next_state and next_opener:
                    if remaining_questions:
                        prompt_parts.append(f"""[대화 지침]
너의 최우선 목표는 사용자의 마지막 말에 공감하고 '꼬리 질문'을 하는 거야. **오직 하나의 질문만** 해.

[현재 상태 목표]
지금은 '{current_goal}'에 대한 정보를 얻는 중이야.

[참고 질문들 - 방향성만 참고하고 절대 그대로 말하지 마]
{chr(10).join([f"- {q}" for q in remaining_questions[:2]])}

[화제 전환 옵션]
하지만, 만약 꼬리 질문할 게 마땅치 않거나, 
현재 주제({current_goal})에 대해 얘기가 충분히 된 것 같다고 판단되면,
아래 '다음 주제 오프너'를 사용해서 자연스럽게 대화를 다음 단계로 넘어가.

[다음 주제 오프너]
"{next_opener}"
""")
                    else:
                        prompt_parts.append(f"""[대화 지침]
너의 최우선 목표는 사용자의 마지막 말에 공감하고 '꼬리 질문'을 하는 거야. **오직 하나의 질문만** 해.

[화제 전환 옵션]
하지만, 만약 꼬리 질문할 게 마땅치 않거나, 
현재 주제({current_goal})에 대해 얘기가 충분히 된 것 같다고 판단되면,
아래 '다음 주제 오프너'를 사용해서 자연스럽게 대화를 다음 단계로 넘어가.

[다음 주제 오프너]
"{next_opener}"
""")
                else:
                    prompt_parts.append(f"[대화 지침]\n사용자의 마지막 말에 공감하고 '꼬리 질문'을 해. **오직 하나의 질문만** 해. 현재 주제({current_goal})에 대해 더 깊이 들어가.")
        
        # 기본 지시사항 (특별한 경우가 아닐 때)
        elif not special_instruction:
            prompt_parts.append("[대화 지침]\n사용자의 마지막 답변에 공감하고, 그 답변 내용 중에서 더 궁금한 점이나 깊게 파고들 부분을 자연스럽게 질문해. **오직 하나의 질문만** 해.")
        
        # 특별 지시사항 추가 (브릿지, redirect 등)
        if special_instruction:
            prompt_parts.append(special_instruction.strip())
        
        # 사용자 메시지 추가
        prompt_parts.append(f"{username}: {user_message}")
        
        return "\n".join(prompt_parts)
    
    
    def generate_response(self, user_message: str, username: str = "사용자") -> dict:
        """
        사용자 메시지에 대한 응답을 생성합니다.
        ResponseGenerator에 위임합니다.
        
        Args:
            user_message: 사용자 메시지
            username: 사용자 이름
            
        Returns:
            {'reply': str, 'image': str} 형태의 딕셔너리
        """
        return self.response_generator.generate_response(user_message, username)


# ============================================================================
# 싱글톤 패턴
# ============================================================================

_chatbot_service = None

def get_chatbot_service():
    """챗봇 서비스 인스턴스 반환 (싱글톤)"""
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
