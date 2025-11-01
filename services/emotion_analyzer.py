"""
감정 분석 모듈

연애 감정(미련도) 분석 및 리포트 생성을 담당합니다.
"""
from typing import Dict, List

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_keywords = self._load_emotion_keywords()
    
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
    
    def calculate_regret_index(self, user_message: str) -> Dict[str, float]:
        """
        종합 미련도 지수 계산
        
        Args:
            user_message (str): 사용자 메시지
            
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


class ReportGenerator:
    """
    리포트 생성 클래스
    
    감정 분석 결과를 기반으로 사용자에게 보기 좋은 리포트를 생성합니다.
    """
    
    def __init__(self):
        pass
    
    def generate_emotion_report(self, analysis_results: Dict[str, float], username: str) -> str:
        """
        감정 리포트 생성
        
        Args:
            analysis_results (Dict[str, float]): 감정 분석 결과
            username (str): 사용자 이름
            
        Returns:
            str: 포맷팅된 리포트 문자열
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