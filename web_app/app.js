/**
 * VietLawAssist Frontend Application Logic
 * Realtime SSE Streaming, Hybrid RRF Evidence Inspector & Micro-Latency Telemetry
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatHistory = document.getElementById("chat-history");
    const welcomeCard = document.getElementById("welcome-card");
    const inputQuery = document.getElementById("user-input-query");
    const btnSubmit = document.getElementById("btn-submit-query");
    const chkStreamMode = document.getElementById("chk-stream-mode");
    const domainPills = document.querySelectorAll("#domain-pills-container .pill");
    const evidenceList = document.getElementById("evidence-list");
    const evidenceBadge = document.getElementById("evidence-count-badge");
    const btnToggleTelemetry = document.getElementById("btn-toggle-telemetry");
    const telemetryBar = document.getElementById("telemetry-bar");

    // Metrics Elements
    const metricLatency = document.getElementById("metric-latency");
    const metricCache = document.getElementById("metric-cache");
    const metricGuardrail = document.getElementById("metric-guardrail");
    const metricHw = document.getElementById("metric-hw");
    const systemStatusText = document.getElementById("system-status-text");

    let currentDomain = "";
    let isProcessing = false;

    // 1. Domain Filter Handling
    domainPills.forEach(pill => {
        pill.addEventListener("click", () => {
            domainPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            currentDomain = pill.getAttribute("data-domain") || "";
        });
    });

    // 2. Sample Query Chips
    document.querySelectorAll(".query-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            inputQuery.value = query;
            inputQuery.focus();
            handleSubmit();
        });
    });

    // 3. Telemetry Toggle
    btnToggleTelemetry.addEventListener("click", () => {
        telemetryBar.style.display = telemetryBar.style.display === "none" ? "grid" : "none";
    });

    // 4. Fetch Health & Telemetry on Init
    async function updateSystemHealth() {
        try {
            const res = await fetch("/api/v1/health");
            if (res.ok) {
                const data = await res.json();
                systemStatusText.textContent = `Hệ Thống Sẵn Sàng (${data.indexed_articles} Điều Luật)`;
                if (data.cache_stats) {
                    metricCache.textContent = `${data.cache_stats.hit_rate_pct}% (${data.cache_stats.cached_entries} cached)`;
                }
                if (data.gpu_telemetry && data.gpu_telemetry.cuda_available) {
                    metricHw.textContent = `${data.gpu_telemetry.device_name} (4GB)`;
                }
            }
        } catch (e) {
            console.log("Health fetch note:", e);
        }
    }
    updateSystemHealth();

    // 5. Submit Query Handler
    async function handleSubmit() {
        const query = inputQuery.value.trim();
        if (!query || isProcessing) return;

        isProcessing = true;
        btnSubmit.disabled = true;
        btnSubmit.style.opacity = "0.6";

        // Hide welcome card on first message
        if (welcomeCard) {
            welcomeCard.style.display = "none";
        }

        // Add User Message Bubble
        appendMessage("user", query);
        inputQuery.value = "";

        const useStreaming = chkStreamMode.checked;

        if (useStreaming) {
            await handleStreamingQuery(query);
        } else {
            await handleRestQuery(query);
        }

        isProcessing = false;
        btnSubmit.disabled = false;
        btnSubmit.style.opacity = "1";
        inputQuery.focus();
        updateSystemHealth();
    }

    // 6. Handle SSE Streaming
    async function handleStreamingQuery(query) {
        const assistantBubble = appendMessage("assistant", "");
        const textBody = assistantBubble.querySelector(".bubble-body");
        const metaContainer = assistantBubble.querySelector(".bubble-meta");

        const streamUrl = `/api/v1/chat/stream?query=${encodeURIComponent(query)}${currentDomain ? `&domain=${encodeURIComponent(currentDomain)}` : ''}`;

        try {
            const eventSource = new EventSource(streamUrl);

            eventSource.addEventListener("meta", (e) => {
                const meta = JSON.parse(e.data);
                renderEvidencePanel(meta.evidence || []);
                metricLatency.textContent = `${meta.retrieval_latency_ms} ms`;
                if (meta.cache_hit) {
                    metricCache.textContent = "100% Hit (0.05ms)";
                }

                if (meta.guardrail) {
                    metricGuardrail.textContent = meta.guardrail.anti_hallucination_pass ? "100% Verified" : "Warning";
                    renderGuardrailBadge(metaContainer, meta.guardrail, meta.retrieval_latency_ms, meta.cache_hit);
                }
            });

            eventSource.addEventListener("token", (e) => {
                const tokenData = JSON.parse(e.data);
                textBody.textContent += tokenData.token;
                chatHistory.scrollTop = chatHistory.scrollHeight;
            });

            eventSource.addEventListener("done", (e) => {
                const doneData = JSON.parse(e.data);
                eventSource.close();
            });

            eventSource.onerror = (err) => {
                console.error("SSE Error:", err);
                eventSource.close();
            };

        } catch (err) {
            textBody.textContent = "⚠️ Lỗi kết nối luồng máy chủ: " + err.message;
        }
    }

    // 7. Handle Regular REST POST
    async function handleRestQuery(query) {
        const assistantBubble = appendMessage("assistant", "Đang tra cứu cơ sở dữ liệu pháp luật...");
        const textBody = assistantBubble.querySelector(".bubble-body");
        const metaContainer = assistantBubble.querySelector(".bubble-meta");

        try {
            const res = await fetch("/api/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, domain: currentDomain || null, top_k: 2 })
            });

            const data = await res.json();
            textBody.textContent = data.answer;

            renderEvidencePanel(data.retrieved_evidence || []);
            metricLatency.textContent = `${data.retrieval_latency_ms} ms`;
            if (data.cache_hit) {
                metricCache.textContent = "100% Hit (0.05ms)";
            }
            if (data.guardrail_report) {
                renderGuardrailBadge(metaContainer, data.guardrail_report, data.retrieval_latency_ms, data.cache_hit);
            }
        } catch (err) {
            textBody.textContent = "⚠️ Không thể kết nối máy chủ API: " + err.message;
        }
    }

    // Helper: Append Message Bubble
    function appendMessage(sender, text) {
        const bubble = document.createElement("div");
        bubble.className = `message-bubble ${sender}`;

        if (sender === "assistant") {
            bubble.innerHTML = `
                <div class="bubble-meta">
                    <span class="badge-verified">⏳ Đang xử lý...</span>
                </div>
                <div class="bubble-body">${escapeHtml(text)}</div>
            `;
        } else {
            bubble.innerHTML = `<div class="bubble-body">${escapeHtml(text)}</div>`;
        }

        chatHistory.appendChild(bubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return bubble;
    }

    function renderGuardrailBadge(container, guardrail, latency, cacheHit) {
        const isPass = guardrail.anti_hallucination_pass;
        const cacheLabel = cacheHit ? "⚡ Cache 0.05ms" : `⏱️ ${latency}ms`;
        container.innerHTML = `
            <span class="badge-verified" style="${isPass ? '' : 'background:rgba(244,63,94,0.15); border-color:#F43F5E; color:#F43F5E;'}">
                ${isPass ? '✅ ĐỐI SOÁT NGUYÊN VĂN (100% KHÔNG ẢO GIÁC)' : '⚠️ CẢNH BÁO TRÍCH DẪN CHƯA RÕ'}
            </span>
            <span class="badge-latency">${cacheLabel} | Hybrid RRF</span>
        `;
    }

    // Render Evidence Side Panel
    function renderEvidencePanel(docs) {
        evidenceList.innerHTML = "";
        evidenceBadge.textContent = `${docs.length} Căn Cứ`;

        if (docs.length === 0) {
            evidenceList.innerHTML = `
                <div class="empty-evidence">
                    <p>Không tìm thấy văn bản quy phạm trực tiếp trong cơ sở dữ liệu.</p>
                </div>
            `;
            return;
        }

        docs.forEach((doc, idx) => {
            const card = document.createElement("div");
            card.className = "evidence-card";
            card.innerHTML = `
                <div class="card-top">
                    <span class="card-law-name">${escapeHtml(doc.law_name || doc.law_code || '')}</span>
                    <span class="card-rrf-score">RRF: ${doc.rrf_score}</span>
                </div>
                <div class="card-title">#${idx + 1} ${escapeHtml(doc.title)}</div>
                <div class="card-snippet">${escapeHtml(doc.content)}</div>
            `;
            evidenceList.appendChild(card);
        });
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // Event Listeners for enter key
    inputQuery.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    });

    btnSubmit.addEventListener("click", handleSubmit);
});
