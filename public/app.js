const statusEl = document.querySelector("#status");
const documentsEl = document.querySelector("#documents");
const uploadForm = document.querySelector("#uploadForm");
const askForm = document.querySelector("#askForm");
const answerEl = document.querySelector("#answer");
const citationsEl = document.querySelector("#citations");
const resetBtn = document.querySelector("#resetBtn");

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function renderDocuments(data) {
  statusEl.textContent = `${data.documents.length} 个文档 / ${data.chunks} 个片段`;
  documentsEl.innerHTML = data.documents.length
    ? data.documents.map((doc) => `
        <div class="document">
          <strong>${escapeHtml(doc.name)}</strong>
          <span>${doc.chunks} 个片段 · ${new Date(doc.createdAt).toLocaleString()}</span>
        </div>
      `).join("")
    : `<p class="muted">还没有上传文档。</p>`;
}

async function refreshDocuments() {
  const data = await api("/api/documents");
  renderDocuments(data);
}

function renderCitations(citations) {
  citationsEl.innerHTML = citations.map((item) => `
    <article class="citation">
      <header>
        <span>[${item.rank}] ${escapeHtml(item.documentName)}</span>
        <span>${Math.round(item.score * 100)}%</span>
      </header>
      <p>${escapeHtml(item.preview)}...</p>
    </article>
  `).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;"
  })[char]);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = document.querySelector("#files").files;
  if (!files.length) return;

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  statusEl.textContent = "正在索引文档...";
  await api("/api/upload", { method: "POST", body: formData });
  uploadForm.reset();
  await refreshDocuments();
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#question").value.trim();
  if (!question) return;

  answerEl.textContent = "正在检索并生成答案...";
  citationsEl.innerHTML = "";

  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    answerEl.textContent = `${data.answer}\n\n模式：${data.mode === "llm" ? "大模型生成" : "本地抽取式回答"}`;
    renderCitations(data.citations);
  } catch (error) {
    answerEl.textContent = error.message;
  }
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("确定清空当前知识库索引吗？")) return;
  await api("/api/reset", { method: "POST" });
  citationsEl.innerHTML = "";
  answerEl.textContent = "知识库已清空。";
  await refreshDocuments();
});

refreshDocuments().catch((error) => {
  statusEl.textContent = error.message;
});
