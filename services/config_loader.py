"""
설정 로더 모듈

설정 파일 및 환경 변수를 로드합니다.
"""
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent


class ConfigLoader:
    
    @staticmethod
    def load_config():
        """
        설정 파일 로드
        
        Returns:
            dict: 챗봇 설정 정보
                - name: 챗봇 이름
                - description: 챗봇 설명
                - system_prompt: 시스템 프롬프트 설정
        
        예시 반환값:
        {
            "name": "환승연애 PD 친구",
            "description": "환승연애팀 막내 PD 친구",
            "system_prompt": {
                "base": "당신은 환승연애팀 막내 PD가 된 친구입니다.",
                "rules": ["친근하게 대화하세요", "연애 이야기를 자연스럽게 이끌어내세요"]
            }
        }
        """
        config_path = BASE_DIR / "config" / "chatbot_config.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] 설정 파일을 찾을 수 없습니다: {config_path}")
            # 기본 설정 반환
            return {
                "name": "환승연애 PD 친구",
                "description": "환승연애팀 막내 PD 친구",
                "system_prompt": {
                    "base": "당신은 환승연애팀 막내 PD가 된 친구입니다.",
                    "rules": ["친근하게 대화하세요", "연애 이야기를 자연스럽게 이끌어내세요"]
                }
            }
