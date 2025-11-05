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

    // 응답 파싱
    // 백엔드에서 {reply: "...", image: "..."} 형태로 반환됨
    const replyText = data.reply || "";
    const imagePath = data.image || null;

    // 디버깅용 로그
    console.log("[DEBUG] API 응답:", { replyText: replyText.substring(0, 50), imagePath });

    appendMessage("bot", replyText, imagePath);

    // needs_report_generation 플래그가 있으면 자동으로 리포트 생성 요청
    if (data.needs_report_generation === true) {
      console.log("[DEBUG] 리포트 생성 자동 요청");
      // 짧은 딜레이 후 자동으로 리포트 생성 요청
      setTimeout(() => {
        // 자동 리포트 요청 (사용자 입력 없이)
        const loadingId2 = appendMessage("bot", "생각 중...");
        
        fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: "", // 빈 메시지로 리포트 생성 트리거
            username: username,
          }),
        })
        .then(response => response.json())
        .then(data => {
          removeMessage(loadingId2);
          const replyText = data.reply || "";
          const imagePath = data.image || null;
          appendMessage("bot", replyText, imagePath);
        })
        .catch(err => {
          console.error("리포트 생성 에러:", err);
          removeMessage(loadingId2);
          appendMessage("bot", "죄송합니다. 리포트 생성 중 오류가 발생했습니다.");
        });
      }, 500);
    }
  } catch (err) {
    console.error("메시지 전송 에러:", err);
    removeMessage(loadingId);
    appendMessage("bot", "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.");
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
    // 이미지가 있으면 먼저 표시
    if (imageSrc) {
      console.log("[DEBUG] 이미지 추가 중:", imageSrc);
      const botImg = document.createElement("img");
      botImg.classList.add("bot-big-img");
      botImg.src = imageSrc;
      botImg.alt = "챗봇 이미지";
      
      // 이미지 로드 에러 처리
      botImg.onerror = function() {
        console.error("[ERROR] 이미지 로드 실패:", imageSrc);
        this.style.display = 'none';
      };
      
      botImg.onload = function() {
        console.log("[DEBUG] 이미지 로드 성공:", imageSrc);
      };
      
      messageElem.appendChild(botImg);
    } else {
      console.log("[DEBUG] 이미지 없음 (imageSrc가 null 또는 undefined)");
    }

    // 텍스트 추가
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
