"""
RAG (Retrieval-Augmented Generation) 서비스

ChromaDB 벡터 검색 및 임베딩 생성을 담당합니다.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import chromadb
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent


class RAGService:
    """
    RAG 서비스 클래스
    
    ChromaDB를 활용한 벡터 검색과 OpenAI 임베딩 생성을 담당합니다.
    """
    
    def __init__(self, openai_client: OpenAI):
        """
        RAG 서비스 초기화
        
        Args:
            openai_client (OpenAI): OpenAI 클라이언트 인스턴스
        """
        self.client = openai_client
        self.collection = self._init_chromadb()
    
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
        
        Returns:
            Collection 또는 None: ChromaDB 컬렉션 객체
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
    
    def create_embedding(self, text: str) -> list:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str): 임베딩할 텍스트
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
        
        Returns:
            list: 3072차원 벡터 (text-embedding-3-large 모델)
                실패 시 빈 리스트 반환
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
    
    def search_similar(self, query: str, threshold: float = 0.45, top_k: int = 5):
        """
        RAG 검색: 유사한 문서 찾기 (핵심 메서드!)
        
        Args:
            query (str): 검색 질의
            threshold (float): 유사도 임계값 (0.3-0.5 권장)
            top_k (int): 검색할 문서 개수
        
        Returns:
            tuple: (document, similarity, metadata) 또는 (None, None, None)
                - document: 가장 유사한 문서 내용
                - similarity: 유사도 점수 (0-1)
                - metadata: 문서 메타데이터
        
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
            
            query_embedding = self.create_embedding(query)
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