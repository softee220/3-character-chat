# 🎯 환승연애 PD 친구 챗봇 구현 가이드

## 📋 구현 완료 사항

### ✅ 1. 핵심 AI 로직 구현 (`services/chatbot_service.py`)

#### 초기화 메서드
- **OpenAI Client**: GPT-4o-mini 모델 사용
- **ChromaDB**: 벡터 데이터베이스 연결
- **LangChain Memory**: 대화 기록 관리
- **감정 분석 키워드**: 연애 감정 분석을 위한 키워드 로드

#### 연애 감정 분석 메서드
- **애착도 분석** (`_analyze_attachment_level`): 0-100점
- **후회도 분석** (`_analyze_regret_level`): 0-100점  
- **미해결감 분석** (`_analyze_unresolved_feelings`): 0-100점
- **비교 기준 분석** (`_analyze_comparison_standard`): 0-100점
- **회피/접근 분석** (`_analyze_avoidance_approach`): 0-100점

#### 미련도 계산 및 리포트 생성
- **종합 미련도 지수**: 가중치 적용 (애착도 30%, 후회도 25%, 미해결감 20%, 비교기준 15%, 회피/접근 10%)
- **감정 리포트**: 5단계 미련도 지수별 개인화된 해석
- **RAG 검색**: 연애 관련 데이터베이스에서 유사한 패턴 검색

### ✅ 2. 연애 분석 데이터 준비 (`static/data/chatbot/chardb_text/`)

#### 데이터 파일들
- **relationship_patterns.txt**: 연애 패턴 분석 데이터
- **emotion_keywords.txt**: 감정 키워드 데이터
- **analysis_templates.txt**: 분석 템플릿 데이터
- **conversation_flows.txt**: 대화 흐름 가이드
- **emotion_analysis_methods.txt**: 감정 분석 방법론
- **response_templates.txt**: 응답 템플릿 데이터
- **emotion_report_examples.txt**: 감정 리포트 예시

### ✅ 3. 설정 파일 업데이트 (`config/chatbot_config.json`)

```json
{
  "name": "환승연애 PD 친구",
  "description": "최근에 환승연애팀 막내 PD가 된 친구입니다...",
  "tags": ["#환승연애", "#PD", "#연애분석", "#미련도측정"],
  "system_prompt": {
    "base": "당신은 환승연애팀 막내 PD가 된 친구입니다...",
    "rules": ["친근하고 호기심 많은 PD 친구처럼 대화하세요", ...]
  }
}
```

### ✅ 4. ChromaDB 구축 스크립트 (`build_chromadb.py`)

- 텍스트 파일 자동 로드
- OpenAI API를 통한 임베딩 생성
- ChromaDB에 벡터 데이터 저장

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 1. 필요한 패키지 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 2. ChromaDB 구축

```bash
# 연애 분석 데이터를 ChromaDB에 저장
python build_chromadb.py
```

### 3. Docker 환경 실행

```bash
# Docker 컨테이너 빌드 및 실행
docker compose up --build
```

### 4. 웹 브라우저에서 접속

```
http://localhost:5001
```

## 🎭 챗봇 사용 방법

### 1. 초기 대화
- "init" 메시지를 보내면 PD 친구가 인사합니다
- 환승연애 프로그램 기획 상황을 설명합니다
- AI 연애 분석 에이전트를 소개합니다

### 2. 연애 이야기 수집
- 자연스러운 질문으로 연애 에피소드를 유도합니다
- 감정적 반응을 관찰하고 키워드를 추출합니다
- 구체적인 상황과 감정을 파악합니다

### 3. 미련도 분석
- 5가지 요소별로 감정을 분석합니다
- 실시간으로 미련도 지수를 계산합니다
- 개인화된 해석을 제공합니다

### 4. 감정 리포트 생성
- "분석", "리포트", "결과" 등의 키워드가 포함된 메시지에 반응
- 종합적인 감정 리포트를 생성합니다
- 미련도 지수별 맞춤 조언을 제공합니다

## 🔧 핵심 기능

### 1. 실시간 감정 분석
```python
# 사용자 메시지에서 감정 분석
analysis_results = self._calculate_regret_index(user_message)
# 결과: {"total": 65.5, "attachment": 70, "regret": 60, ...}
```

### 2. 개인화된 감정 리포트
```python
# 미련도 지수별 해석
if total <= 20:
    level = "완전 정리 단계"
    emoji = "💚"
elif total <= 40:
    level = "잔잔한 여운 단계"
    emoji = "💛"
# ...
```

### 3. RAG 기반 컨텍스트 검색
```python
# 연애 관련 데이터베이스에서 유사한 패턴 검색
context, similarity, metadata = self._search_similar(
    query=user_message,
    threshold=0.45,
    top_k=5
)
```

## 📊 미련도 지수 해석

| 지수 | 단계 | 설명 | 이모지 |
|------|------|------|--------|
| 0-20% | 완전 정리 단계 | 과거를 아름답게 정리하고 새로운 시작 준비 | 💚 |
| 21-40% | 잔잔한 여운 단계 | '그 사람'보다는 '그때의 나'를 그리워 | 💛 |
| 41-60% | 적당한 미련 단계 | 감정이 남아있지만 새로운 시작 준비됨 | 🧡 |
| 61-80% | 강한 미련 단계 | 새로운 관계 시작하기에는 시간 필요 | ❤️ |
| 81-100% | 매우 강한 미련 단계 | 완전한 정리가 필요한 상태 | 💔 |

## 🎨 확장 가능한 기능

### 1. 이미지 분석
- 연애 관련 이미지 업로드 시 감정 분석
- 시각적 요소를 통한 감정 상태 파악

### 2. 음성 분석
- 음성 톤을 통한 감정 강도 측정
- 말하는 속도와 톤 변화 분석

### 3. 대화 패턴 분석
- 대화 길이와 깊이 분석
- 감정 변화 추이 모니터링

### 4. 개인화된 조언
- 사용자별 맞춤 조언 생성
- 단계별 치유 가이드 제공

## 🐛 디버깅 팁

### 1. 로그 확인
```python
print(f"[ANALYSIS] 미련도: {analysis_results['total']:.1f}%")
print(f"[RAG] Context found: {has_context}")
```

### 2. ChromaDB 상태 확인
```python
# 컬렉션 문서 수 확인
print(f"[INFO] 문서 수: {collection.count()}")
```

### 3. API 호출 상태 확인
```python
print(f"[LLM] Calling API...")
print(f"[BOT] {reply[:100]}...")
```

## 📚 참고 자료

- [OpenAI API 문서](https://platform.openai.com/docs)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [LangChain 문서](https://python.langchain.com/)
- [Flask 문서](https://flask.palletsprojects.com/)

## 🎯 다음 단계

1. **이미지 추가**: PD 친구 캐릭터 이미지 및 분석 차트 이미지
2. **성능 최적화**: RAG 검색 속도 개선
3. **UI 개선**: 감정 리포트 시각화
4. **배포**: Vercel을 통한 프로덕션 배포

