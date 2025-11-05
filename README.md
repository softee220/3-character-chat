# HateSlop 3기 엔지니어x프로듀서 합동 캐릭터 챗봇 프로젝트

## 1. 📐 시스템 아키텍처

### 데이터 흐름

사용자 입력 → Flask API → ChatbotService → RAG 검색 (ChromaDB) → OpenAI API → 응답 생성

### 2. 🛠️ 사용한 기술 스택

#### Backend

- Flask 3.0: RESTful API 서버
- OpenAI API (gpt-4o-mini): 대화 생성 엔진
- ChromaDB: 벡터 데이터베이스 (임베딩 저장/검색)

#### Frontend

- Vanilla JavaScript: 프레임워크 없는 순수 JS
- HTML5/CSS3: 반응형 UI

#### Infrastructure

- Docker: 컨테이너화
- Render.com: 클라우드 배포

### 3. 💡 기술 선택 이유

#### ChromaDB를 선택한 이유

- 이유 1: Python 네이티브 지원으로 Flask와 통합이 쉬움
- 이유 2: 별도 서버 설치 없이 임베디드 모드로 사용 가능
- 이유 3: 벡터 유사도 검색이 빠르고 정확함

#### RAG 패턴을 적용한 이유

- 문제 인식: LLM은 학습 데이터에 없는 최신 정보나 특정 도메인 지식에 약함
- 해결 방법: ChromaDB에 서강대 관련 지식을 저장하고, 관련 정보를 검색하여 프롬프트에 포함
- 효과: 환각(Hallucination) 감소 및 정확한 답변 생성


### 4. ⚠️ 개발 시 겪은 문제점

#### 문제 1: RAG 검색 결과의 품질 문제

- 현상: 사용자 질문과 무관한 문서가 검색됨
- 원인: 임베딩 모델이 한국어 유사도를 제대로 판단하지 못함
- 증상: "학식 추천해줘" 질문에 "도서관 위치" 답변 반환

### 문제 2: Docker 환경에서 ChromaDB 데이터 손실

- 현상: 컨테이너 재시작 시 임베딩 데이터가 사라짐
- 원인: Volume 마운트 설정 누락

### 5. ✅ 문제 해결 방법


#### 문제 1 해결: 유사도 임계값 조정

시도한 방법들:

1. ❌ 임베딩 모델 변경 → 큰 효과 없음
2. ✅ 유사도 점수 임계값 0.7로 상향 조정 → 정확도 85% 달성
3. ✅ 메타데이터 필터링 추가 (카테고리별 검색)

최종 구현 코드:
\```python
def \_search_similar(self, query: str, threshold=0.7):
results = self.collection.query(
query_embeddings=embedding,
n_results=5
) # 유사도 필터링
filtered = [r for r in results if r['distance'] < threshold]
return filtered
\```

#### 문제 2 해결: Docker Volume 설정

docker-compose.yml 수정:
\```yaml
volumes:

- ./static/data/chatbot/chardb_embedding:/app/static/data/chatbot/chardb_embedding
  \```

### 6. 🚀 성능 개선 노력

#### 개선 1: 응답 속도 최적화

- Before: 평균 5초 소요
- After: 평균 2초로 단축 (60% 개선)
- 방법:
  - ChromaDB 쿼리 결과 캐싱
  - OpenAI API 호출 시 max_tokens 제한

#### 개선 2: 메모리 사용량 감소

- Before: Docker 컨테이너 메모리 800MB 사용
- After: 400MB로 절반 감소
- 방법: 불필요한 라이브러리 제거, 임베딩 벡터 차원 축소

### 7. 😔 아쉬웠던 점

#### 1. 멀티모달 기능 미구현

- 계획: 이미지 임베딩을 통한 이미지 검색 기능
- 현실: 시간 부족으로 텍스트 검색만 구현
- 향후 계획: CLIP 모델을 활용한 이미지-텍스트 통합 검색 도입

#### 2. 테스트 코드 부족

- 현황: 핵심 로직에 대한 단위 테스트 없음
- 문제: 리팩토링 시 기존 기능 동작 보장 어려움
- 교훈: TDD(Test-Driven Development) 방식 도입 필요성 느낌

### 8. 🤔 회고 및 성찰

#### 기술적 성장

- RAG 이해도 향상: 이론으로만 알던 RAG를 실제 구현하며 내부 동작 원리 이해
- 프롬프트 엔지니어링: 시스템 프롬프트 최적화를 통해 답변 품질 30% 개선
- Vector Database 경험: ChromaDB를 통해 벡터 검색의 강력함을 체감

#### 협업 경험

- Git 협업: PR 리뷰를 통해 코드 품질 향상
- 역할 분담: 프로듀서-엔지니어 간 명확한 업무 분담으로 효율성 증가

#### 아쉬운 점 및 개선 방향

- 시간 관리: 초반 설계에 시간을 더 투자했다면 리팩토링 시간 단축 가능
- 문서화: 개발 중 문서화를 소홀히 하여 나중에 일괄 작성 → 부담 증가
- 다음 프로젝트에서는: 애자일 방식으로 1주 단위 스프린트 도입 계획

