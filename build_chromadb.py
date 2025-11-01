#!/usr/bin/env python3
"""
ChromaDB 데이터베이스 구축 스크립트

이 스크립트는 static/data/chatbot/chardb_text/ 폴더의 텍스트 파일들을
ChromaDB에 임베딩하여 저장합니다.

실행 방법:
python build_chromadb.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent

def load_text_files():
    """chardb_text 폴더의 모든 텍스트 파일을 로드"""
    text_dir = BASE_DIR / "static" / "data" / "chatbot" / "chardb_text"
    
    if not text_dir.exists():
        print(f"[ERROR] 텍스트 폴더를 찾을 수 없습니다: {text_dir}")
        return []
    
    documents = []
    metadatas = []
    ids = []
    
    for file_path in text_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            documents.append(content)
            metadatas.append({
                "filename": file_path.name,
                "type": "relationship_analysis",
                "source": "chardb_text"
            })
            ids.append(f"doc_{len(documents)}")
            
            print(f"[LOADED] {file_path.name} ({len(content)} chars)")
            
        except Exception as e:
            print(f"[ERROR] 파일 로드 실패 {file_path}: {e}")
    
    return documents, metadatas, ids

def create_embeddings(documents):
    """문서들을 임베딩으로 변환"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    
    client = OpenAI(api_key=api_key)
    embeddings = []
    
    for i, doc in enumerate(documents):
        try:
            response = client.embeddings.create(
                input=[doc],
                model="text-embedding-3-large"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            print(f"[EMBEDDING] 문서 {i+1}/{len(documents)} 완료")
            
        except Exception as e:
            print(f"[ERROR] 임베딩 생성 실패 (문서 {i+1}): {e}")
            return []
    
    return embeddings

def build_chromadb():
    """ChromaDB 데이터베이스 구축"""
    print("=" * 60)
    print("ChromaDB 데이터베이스 구축 시작")
    print("=" * 60)
    
    # 1. 텍스트 파일 로드
    print("\n[1단계] 텍스트 파일 로드")
    documents, metadatas, ids = load_text_files()
    
    if not documents:
        print("[ERROR] 로드할 문서가 없습니다.")
        return False
    
    print(f"[SUCCESS] {len(documents)}개 문서 로드 완료")
    
    # 2. 임베딩 생성
    print("\n[2단계] 임베딩 생성")
    embeddings = create_embeddings(documents)
    
    if not embeddings:
        print("[ERROR] 임베딩 생성 실패")
        return False
    
    print(f"[SUCCESS] {len(embeddings)}개 임베딩 생성 완료")
    
    # 3. ChromaDB 저장
    print("\n[3단계] ChromaDB 저장")
    
    db_path = BASE_DIR / "static" / "data" / "chatbot" / "chardb_embedding"
    db_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(path=str(db_path))
        
        # 기존 컬렉션 삭제 (있다면)
        try:
            client.delete_collection("rag_collection")
            print("[INFO] 기존 컬렉션 삭제")
        except:
            pass
        
        # 새 컬렉션 생성
        collection = client.create_collection(
            name="rag_collection",
            metadata={"description": "연애 감정 분석 데이터"}
        )
        
        # 데이터 추가
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[SUCCESS] ChromaDB 저장 완료: {db_path}")
        print(f"[INFO] 컬렉션: {collection.name}")
        print(f"[INFO] 문서 수: {collection.count()}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] ChromaDB 저장 실패: {e}")
        return False

def main():
    """메인 함수"""
    try:
        success = build_chromadb()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ChromaDB 구축 완료!")
            print("이제 챗봇을 실행할 수 있습니다.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ ChromaDB 구축 실패!")
            print("오류를 확인하고 다시 시도해주세요.")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n[ERROR] 스크립트 실행 실패: {e}")

if __name__ == "__main__":
    main()

