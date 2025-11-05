console.log("챗봇 JS 로드 완료");

// DOM 요소
const chatArea = document.querySelector(".chat-area");
const username = chatArea ? chatArea.dataset.username : "사용자";
const chatLog = document.getElementById("chat-log");
const userMessageInput = document.getElementById("user-message");
const sendBtn = document.getElementById("send-btn");
const videoBtn = document.getElementById("videoBtn");
const imageBtn = document.getElementById("imageBtn");

// 메시지 전송 함수
async function sendMessage(isInitial = false) {
  let message;

  if (isInitial) {
    message = "init";
  } else {
    message = userMessageInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    userMessageInput.value = "";
  }

  // 로딩 표시
  const loadingId = appendMessage("bot", "생각 중...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        username: username,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // 로딩 메시지 제거
    removeMessage(loadingId);

    // 프로그램 추천 응답인지 확인
    if (data.program_recommendation) {
      // 프로그램 추천 페이지로 즉시 리다이렉트
      const recommendation = data.program_recommendation;
      const params = new URLSearchParams({
        image: recommendation.image || data.image || '',
        message: recommendation.message || data.reply || '',
        sentiment: recommendation.sentiment || 'positive'
      });
      
      // 바로 리다이렉트
      window.location.href = `/program_recommendation?${params.toString()}`;
      
      return;
    }

    // 응답 파싱
    // 백엔드에서 {reply: "...", image: "..."} 형태로 반환됨
    const replyText = data.reply || "";
    const imagePath = data.image || null;

    // 디버깅용 로그
    console.log("[DEBUG] API 응답:", { replyText: replyText.substring(0, 50), imagePath });

    appendMessage("bot", replyText, imagePath);
  } catch (err) {
    console.error("메시지 전송 에러:", err);
    removeMessage(loadingId);
    appendMessage("bot", "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.");
  }
}

// 이미지 업데이트 함수
function updateCharacterImage(imageSrc) {
  const characterImage = document.getElementById("character-image");
  const characterName = document.getElementById("character-name");
  
  if (characterImage && imageSrc) {
    console.log("[DEBUG] 캐릭터 이미지 업데이트:", imageSrc);
    characterImage.src = imageSrc;
    characterImage.classList.add("show");
    
    // 캐릭터 이름도 표시
    if (characterName) {
      characterName.classList.add("show");
    }
    
    // 이미지 로드 에러 처리
    characterImage.onerror = function() {
      console.error("[ERROR] 이미지 로드 실패:", imageSrc);
      this.style.display = 'none';
      if (characterName) {
        characterName.classList.remove("show");
      }
    };
    
    characterImage.onload = function() {
      console.log("[DEBUG] 이미지 로드 성공:", imageSrc);
      this.style.display = 'block';
      if (characterName) {
        characterName.classList.add("show");
      }
    };
  }
}

// 메시지 DOM에 추가
let messageIdCounter = 0;
function appendMessage(sender, text, imageSrc = null) {
  const messageId = `msg-${messageIdCounter++}`;
  const messageElem = document.createElement("div");
  messageElem.classList.add("message", sender);
  messageElem.id = messageId;

  if (sender === "user") {
    messageElem.textContent = text;
  } else {
    // 이미지가 있으면 왼쪽 패널에 표시
    if (imageSrc) {
      updateCharacterImage(imageSrc);
    }

    // 텍스트만 메시지에 추가
    const textContainer = document.createElement("div");
    textContainer.classList.add("bot-text-container");
    textContainer.textContent = text;
    messageElem.appendChild(textContainer);
  }

  if (chatLog) {
    chatLog.appendChild(messageElem);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  return messageId;
}

// 메시지 제거
function removeMessage(messageId) {
  const elem = document.getElementById(messageId);
  if (elem) {
    elem.remove();
  }
}

// 엔터키로 전송
if (userMessageInput) {
  userMessageInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  });
}

// 전송 버튼
if (sendBtn) {
  sendBtn.addEventListener("click", () => sendMessage());
}

// 모달 열기/닫기
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "block";
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "none";
  }
}

// 미디어 버튼 이벤트
if (videoBtn) {
  videoBtn.addEventListener("click", () => openModal("videoModal"));
}

if (imageBtn) {
  imageBtn.addEventListener("click", () => openModal("imageModal"));
}

// 모달 닫기 버튼
document.querySelectorAll(".modal-close").forEach((btn) => {
  btn.addEventListener("click", () => {
    const modalId = btn.dataset.closeModal;
    closeModal(modalId);
  });
});

// 모달 배경 클릭 시 닫기
document.querySelectorAll(".modal").forEach((modal) => {
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      modal.style.display = "none";
    }
  });
});

// 페이지 로드 시 초기 메시지 요청
window.addEventListener("load", () => {
  console.log("페이지 로드 완료");

  setTimeout(() => {
    if (chatLog && chatLog.childElementCount === 0) {
      console.log("초기 메시지 요청");
      sendMessage(true);
    }
  }, 500);
});
