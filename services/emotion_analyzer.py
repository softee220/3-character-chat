"""
감정 분석 모듈

연애 감정(미련도) 분석 및 리포트 생성을 담당합니다.
"""
from typing import Dict, List, Optional, Any
import json

class EmotionAnalyzer:
    def __init__(self, rag_service=None, openai_client=None):
        """
        Args:
            rag_service: RAGService 인스턴스 (옵션)
            openai_client: OpenAI 클라이언트 (옵션)
        """
        self.emotion_keywords = self._load_emotion_keywords()
        self.rag_service = rag_service
        self.openai_client = openai_client
    
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
    
    def calculate_regret_index(self, user_message: str, use_rag: bool = True) -> Dict[str, float]:
        """
        종합 미련도 지수 계산
        
        Args:
            user_message (str): 사용자 메시지
            use_rag (bool): RAG 기반 정규화 사용 여부
            
        Returns:
            Dict[str, float]: {
                'total': float,      # 종합 미련도 지수
                'attachment': float,  # 애착도
                'regret': float,      # 후회도
                'unresolved': float,  # 미해결감
                'comparison': float,  # 비교 기준
                'avoidance': float    # 회피/접근
            }
        """
        # 기본 키워드 기반 분석
        attachment = self._analyze_attachment_level(user_message)
        regret = self._analyze_regret_level(user_message)
        unresolved = self._analyze_unresolved_feelings(user_message)
        comparison = self._analyze_comparison_standard(user_message)
        avoidance = self._analyze_avoidance_approach(user_message)
        
        # RAG 기반 정규화 (옵션)
        if use_rag and self.rag_service and self.openai_client:
            try:
                normalized_scores = self._normalize_with_rag(user_message, {
                    'attachment': attachment,
                    'regret': regret,
                    'unresolved': unresolved,
                    'comparison': comparison,
                    'avoidance': avoidance
                })
                attachment = normalized_scores['attachment']
                regret = normalized_scores['regret']
                unresolved = normalized_scores['unresolved']
                comparison = normalized_scores['comparison']
                avoidance = normalized_scores['avoidance']
                print("[ANALYSIS] RAG 기반 정규화 적용 완료")
            except Exception as e:
                print(f"[WARNING] RAG 정규화 실패, 기본 분석 결과 사용: {e}")
        
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
    
    def _normalize_with_rag(self, user_message: str, initial_scores: Dict[str, float]) -> Dict[str, float]:
        """
        RAG 기반 미련도 정규화 (LLM-as-a-Grader)
        
        Args:
            user_message (str): 사용자 답변 전체
            initial_scores (Dict[str, float]): 초기 키워드 기반 점수
        
        Returns:
            Dict[str, float]: 정규화된 점수
        """
        # RAG로 유사 사례 검색
        similar_cases = self.rag_service.search_similar_cases(user_message, top_k=3)
        
        if not similar_cases:
            print("[ANALYSIS] 유사 사례 없음, 기본 분석 결과 반환")
            return initial_scores
        
        # LLM-as-a-Grader 프롬프트 구성
        prompt = self._build_llm_grader_prompt(user_message, initial_scores, similar_cases)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-pro",
                messages=[
                    {"role": "system", "content": "당신은 연애 감정 분석 전문가입니다. 사용자의 답변을 정확하게 분석하여 1-100점 사이로 평가하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # JSON 응답에서 점수 추출
            normalized = {
                'attachment': result.get('attachment', initial_scores['attachment']),
                'regret': result.get('regret', initial_scores['regret']),
                'unresolved': result.get('unresolved', initial_scores['unresolved']),
                'comparison': result.get('comparison', initial_scores['comparison']),
                'avoidance': result.get('avoidance', initial_scores['avoidance'])
            }
            
            print(f"[ANALYSIS] LLM 정규화 완료: {normalized}")
            return normalized
            
        except Exception as e:
            print(f"[ERROR] LLM 정규화 실패: {e}")
            return initial_scores
    
    def _build_llm_grader_prompt(self, user_message: str, initial_scores: Dict[str, float], cases: List[Dict]) -> str:
        """
        LLM-as-a-Grader 프롬프트 구성
        
        Args:
            user_message (str): 사용자 답변
            initial_scores (Dict[str, float]): 초기 점수
            cases (List[Dict]): 유사 사례들
        
        Returns:
            str: 프롬프트
        """
        # 유사 사례 요약
        cases_text = ""
        for i, case in enumerate(cases, 1):
            case_score = case.get('analysis', {}).get('score', 0)
            keywords = ' '.join(case.get('analysis', {}).get('keywords', []))
            cases_text += f"\n[사례 {i}] (종합 미련도: {case_score}%)\n"
            cases_text += f"요약: {case.get('summary', '')}\n"
            cases_text += f"키워드: {keywords}\n"
            cases_text += f"- 애착도: {case.get('analysis', {}).get('attachment', {}).get('score', 0)}% - {case.get('analysis', {}).get('attachment', {}).get('reason', '')}\n"
            cases_text += f"- 후회도: {case.get('analysis', {}).get('regret', {}).get('score', 0)}% - {case.get('analysis', {}).get('regret', {}).get('reason', '')}\n"
            cases_text += f"- 미해결감: {case.get('analysis', {}).get('unresolved', {}).get('score', 0)}% - {case.get('analysis', {}).get('unresolved', {}).get('reason', '')}\n"
            cases_text += f"- 비교 기준: {case.get('analysis', {}).get('comparison', {}).get('score', 0)}% - {case.get('analysis', {}).get('comparison', {}).get('reason', '')}\n"
            cases_text += f"- 회피/접근: {case.get('analysis', {}).get('avoidance', {}).get('score', 0)}% - {case.get('analysis', {}).get('avoidance', {}).get('reason', '')}\n"
        
        prompt = f"""다음은 사용자의 연애 미련도 분석을 위한 답변입니다.

**[사용자 답변]**
{user_message}

**[참고 사례 및 분석 기준]**
{cases_text}

위 사례들을 참고하여 다음 5가지 기준으로 사용자 답변을 **1점에서 100점 사이**로 평가하고, 각 지표에 대한 **근거를 1문장**으로 작성하세요.

**5가지 평가 기준:**
1. **애착도 (attachment)**: 아직도 그 사람에 대한 감정적 유대감
2. **후회도 (regret)**: 그때 더 잘했어야 했다는 자책감
3. **미해결감 (unresolved)**: 명확한 결론 없이 끝난 상태
4. **비교 기준 (comparison)**: 전 연인을 이상화하여 비교하게 되는 정도
5. **회피/접근 (avoidance)**: 그 사람을 피하거나 만나고 싶은 욕구

JSON 형식으로만 응답하세요:

```json
{{
    "attachment": 75,
    "attachment_reason": "SNS를 자주 확인하고 마음이 흔들린다는 표현에서 강한 애착 감지",
    "regret": 85,
    "regret_reason": "연락을 피한 것과 더 잘했어야 했다는 명확한 후회",
    "unresolved": 80,
    "unresolved_reason": "작은 오해로 인한 미해결 상태",
    "comparison": 60,
    "comparison_reason": "과거를 곱씹는 행동은 비교 기준 형성 징후",
    "avoidance": 70,
    "avoidance_reason": "연락하고 싶지만 참고 있다는 강한 회피 노력"
}}
```

**중요:** 참고 사례의 분석 근거를 **직접 참고**하여 사용자 답변에 적용하세요. 점수는 참고 사례와 유사한 맥락으로 평가하되, 사용자의 실제 표현을 정확히 반영하세요."""
        
        return prompt


class ReportGenerator:
    """
    리포트 생성 클래스
    
    감정 분석 결과를 기반으로 사용자에게 보기 좋은 리포트를 생성합니다.
    """
    
    def __init__(self, rag_service=None, openai_client=None):
        """
        Args:
            rag_service: RAGService 인스턴스 (옵션)
            openai_client: OpenAI 클라이언트 (옵션)
        """
        self.rag_service = rag_service
        self.openai_client = openai_client
    
    def generate_emotion_report(self, analysis_results: Dict[str, float], username: str, user_message: str = "") -> str:
        """
        감정 리포트 생성 (LLM 기반)
        
        Args:
            analysis_results (Dict[str, float]): 감정 분석 결과
            username (str): 사용자 이름
            user_message (str): 사용자 답변 (전체 대화 맥락)
            
        Returns:
            str: 포맷팅된 리포트 문자열
        """
        # LLM 기반 리포트 생성 시도
        if self.openai_client and self.rag_service and user_message:
            try:
                report = self._generate_llm_report(analysis_results, username, user_message)
                if report:
                    return report
            except Exception as e:
                print(f"[WARNING] LLM 리포트 생성 실패, 기본 리포트 사용: {e}")
        
        # 기본 리포트 생성 (폴백)
        return self._generate_default_report(analysis_results, username)
    
    def _generate_llm_report(self, analysis_results: Dict[str, float], username: str, user_message: str) -> Optional[str]:
        """
        LLM 기반 개인화 리포트 생성
        
        Args:
            analysis_results (Dict[str, float]): 감정 분석 결과
            username (str): 사용자 이름
            user_message (str): 사용자 답변
        
        Returns:
            Optional[str]: LLM 생성 리포트 또는 None
        """
        # RAG로 유사 사례 검색
        similar_cases = self.rag_service.search_similar_cases(user_message, top_k=3)
        
        # 프롬프트 구성
        prompt = self._build_report_prompt(analysis_results, username, user_message, similar_cases)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-pro",
                messages=[
                    {"role": "system", "content": "당신은 '환승연애' PD 친구로, 사용자에게 진심 어린 조언을 하는 따뜻한 친구입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            report = response.choices[0].message.content.strip()
            print("[REPORT] LLM 리포트 생성 완료")
            return report
            
        except Exception as e:
            print(f"[ERROR] LLM 리포트 생성 실패: {e}")
            return None
    
    def _build_report_prompt(self, analysis_results: Dict[str, float], username: str, user_message: str, cases: List[Dict]) -> str:
        """
        리포트 생성을 위한 프롬프트 구성
        
        Args:
            analysis_results (Dict[str, float]): 분석 결과
            username (str): 사용자 이름
            user_message (str): 사용자 답변
            cases (List[Dict]): 유사 사례들
        
        Returns:
            str: 프롬프트
        """
        total = analysis_results.get("total", 0)
        
        # 미련도 단계 결정
        if total <= 20:
            level = "완전 정리 단계"
            emoji = "💚"
        elif total <= 40:
            level = "잔잔한 여운 단계"
            emoji = "💛"
        elif total <= 60:
            level = "적당한 미련 단계"
            emoji = "🧡"
        elif total <= 80:
            level = "강한 미련 단계"
            emoji = "❤️"
        else:
            level = "매우 강한 미련 단계"
            emoji = "💔"
        
        # 유사 사례 요약
        cases_text = ""
        if cases:
            cases_text = "\n[참고: 유사한 다른 사례들]\n"
            for i, case in enumerate(cases[:2], 1):  # 상위 2개만
                cases_text += f"\n사례 {i}: {case.get('summary', '')}\n"
                case_keywords = ' '.join(case.get('analysis', {}).get('keywords', []))
                if case_keywords:
                    cases_text += f"키워드: {case_keywords}\n"
        
        prompt = f"""다음은 {username}님의 연애 미련도 분석 결과입니다.

**[분석 결과]**
- 종합 미련도: {int(total)}% ({level})
- 애착도: {int(analysis_results.get('attachment', 0))}%
- 후회도: {int(analysis_results.get('regret', 0))}%
- 미해결감: {int(analysis_results.get('unresolved', 0))}%
- 비교 기준: {int(analysis_results.get('comparison', 0))}%
- 회피/접근: {int(analysis_results.get('avoidance', 0))}%

**[사용자의 이야기]**
{user_message}
{cases_text}

위 분석 결과를 바탕으로, 친구가 진심으로 응원하는 마음으로 **PD 친구가 직접 작성하는 듯한** 형태의 리포트를 작성하세요.

**리포트 형식:**
```
[{username}님의 연애 감정 리포트]


1️⃣ 주요 감정 키워드
#키워드1 #키워드2 #키워드3


2️⃣ 감정 상태 분석
{level}에 대한 친구가 주는 따뜻한 설명 (2-3문장)


3️⃣ 미련도 지수
{emoji} {int(total)}% — {level}


4️⃣ {username}에게
사용자의 상황과 유사 사례를 참고하여 진심 어린 조언 2-3문장
```

**중요 지침:**
- 친한 친구가 애정 어린 마음으로 조언하는 톤 유지
- "너", "네", "~어", "~해" 같은 반말 사용
- 분석 수치나 딱딱한 표현 지양, 자연스러운 말투
- 사용자의 실제 답변을 반영하여 개인화된 조언
- 유사 사례가 있다면 "같은 경험을 한 사람들도..." 같은 공감 표현 사용
- 희망적이고 따뜻한 마무리
- **각 섹션 사이에는 반드시 빈 줄 1개(\n)를 넣어서 가독성을 높이세요**
- **1번 섹션: 해시태그만 한 줄로 나열 (띄어쓰기로 구분)**
- **2번 섹션: 따옴표 없이 설명만 작성**
- **3번 섹션: 이모지 + 점수 + 단계만 작성 (굵은 글씨 없음)**
- **4번 섹션: "{username}에게" 제목으로 시작, 조언 내용만 작성**

**⚠️ 절대 금지 사항:**
- 리포트 형식 외의 추가 텍스트나 설명을 절대 작성하지 마세요
- "X에 대한 얘기를 더 알려달라", "더 기억나는 게 있어?", "가장 좋았던 순간은?" 같은 질문을 절대 포함하지 마세요
- 리포트 외의 대화나 추가 요청을 절대 포함하지 마세요
- 리포트 형식(1️⃣~4️⃣)만 작성하고, 그 외의 내용은 한 글자도 추가하지 마세요

**리포트 형식에 맞는 내용만 작성하세요. 리포트 외의 텍스트, 질문, 대화는 절대 포함하지 마세요.**"""
        
        return prompt
    
    def _generate_default_report(self, analysis_results: Dict[str, float], username: str) -> str:
        """
        기본 리포트 생성 (LLM 실패 시 폴백)
        
        Args:
            analysis_results (Dict[str, float]): 감정 분석 결과
            username (str): 사용자 이름
            
        Returns:
            str: 기본 리포트 문자열
        """
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
{description}


3️⃣ 미련도 지수
{emoji} {int(total)}% — {level}


4️⃣ {username}에게
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