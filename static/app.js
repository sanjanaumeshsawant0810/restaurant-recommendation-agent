const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const chatLoading = document.getElementById("chat-loading");
const newChatButton = document.getElementById("new-chat-button");
const chatDrawerToggle = document.getElementById("chat-drawer-toggle");
const chatDrawerBackdrop = document.getElementById("chat-drawer-backdrop");
const chatDrawerClose = document.getElementById("chat-drawer-close");
const sessionSidebar = document.getElementById("session-sidebar");
const mapStatus = document.getElementById("map-status");
const agentTrace = document.getElementById("agent-trace");
const traceToggle = document.getElementById("trace-toggle");
const sessionList = document.getElementById("session-list");

let sessionId = null;
let sessions = [];
let map = null;
let markers = [];
let routeLines = [];
let distanceBadges = [];

const userLocationIcon = L.icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function displayText(value) {
  if (typeof value === "string") {
    return value.trim();
  }
  if (value && typeof value === "object") {
    if (typeof value.text === "string") {
      return value.text.trim();
    }
    if (typeof value.label === "string") {
      return value.label.trim();
    }
  }
  return "";
}

function handlePopupImageError(image) {
  if (!image) {
    return;
  }
  const gallery = image.parentElement;
  image.remove();
  if (gallery && !gallery.querySelector("img")) {
    gallery.remove();
    return;
  }
  updatePopupGalleryLayout(gallery);
}

function updatePopupGalleryLayout(gallery) {
  if (!gallery) {
    return;
  }
  const imageCount = gallery.querySelectorAll("img").length;
  gallery.classList.remove("count-1", "count-2", "count-3");
  if (imageCount > 0) {
    gallery.classList.add(`count-${Math.min(imageCount, 3)}`);
  }
}

function setDrawerOpen(isOpen) {
  sessionSidebar.classList.toggle("open", isOpen);
  chatDrawerBackdrop.classList.toggle("open", isOpen);
  chatDrawerToggle.setAttribute("aria-expanded", String(isOpen));
  document.body.classList.toggle("drawer-open", isOpen);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) {
    window.location.href = "/";
    throw new Error(data.error || "Please sign in to continue.");
  }
  if (!response.ok) {
    throw new Error(data.error || data.message || data.reply || `Request failed with status ${response.status}`);
  }
  return data;
}

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const lines = String(text || "").split("\n");
  lines.forEach((line, index) => {
    if (index > 0) {
      node.appendChild(document.createElement("br"));
    }

    const urlMatch = line.match(/^(Website|Google Maps|Directions):\s+(https?:\/\/\S+)$/);
    if (urlMatch) {
      const label = document.createElement("span");
      label.textContent = `${urlMatch[1]}: `;
      node.appendChild(label);

      const link = document.createElement("a");
      link.href = urlMatch[2];
      link.textContent = urlMatch[2];
      link.target = "_blank";
      link.rel = "noreferrer";
      node.appendChild(link);
      return;
    }

    node.appendChild(document.createTextNode(line));
  });
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  return node;
}

function renderMessages(messages = []) {
  chatLog.innerHTML = "";
  messages.forEach((message) => addMessage(message.role, message.text));
}

function setLoadingState(isLoading) {
  if (chatLoading) {
    chatLoading.classList.toggle("visible", isLoading);
    chatLoading.setAttribute("aria-hidden", String(!isLoading));
  }
  chatForm.classList.toggle("is-loading", isLoading);
  chatInput.disabled = isLoading;
  sendButton.disabled = isLoading;
}

function formatSessionTimestamp(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function renderSessionList() {
  sessionList.innerHTML = "";
  if (!sessions.length) {
    sessionList.innerHTML = "<div class='session-empty'>No saved chats yet.</div>";
    return;
  }

  sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = `session-row${session.session_id === sessionId ? " active" : ""}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item${session.session_id === sessionId ? " active" : ""}`;
    button.innerHTML = `
      <div class="session-title">${session.title || "New chat"}</div>
      <div class="session-meta">${formatSessionTimestamp(session.updated_at)}</div>
    `;
    button.addEventListener("click", () => loadSession(session.session_id));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete";
    deleteButton.setAttribute("aria-label", `Delete chat ${session.title || "New chat"}`);
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async () => {
      await deleteSession(session.session_id);
    });

    item.appendChild(button);
    item.appendChild(deleteButton);
    sessionList.appendChild(item);
  });
}

async function deleteSession(targetSessionId) {
  await requestJson(`/api/session/${targetSessionId}`, { method: "DELETE" });
  const deletedCurrentSession = targetSessionId === sessionId;
  await refreshSessions();

  if (!sessions.length) {
    await createSession();
    return;
  }

  if (deletedCurrentSession) {
    await loadSession(sessions[0].session_id);
    return;
  }

  renderSessionList();
}

function renderAgentTrace(traceItems = []) {
  if (!agentTrace) {
    return;
  }
  agentTrace.innerHTML = "";
  if (!traceItems.length) {
    agentTrace.innerHTML = "<div class='trace-item'>The agent trace will appear after the first turn.</div>";
    return;
  }

  traceItems.forEach((item) => {
    const node = document.createElement("div");
    node.className = "trace-item";
    node.innerHTML = `
      <div class="trace-agent">${item.agent}</div>
      <div><span class="trace-action">${item.action}</span>${item.details}</div>
    `;
    agentTrace.appendChild(node);
  });
}

function initMap() {
  map = L.map("map").setView([40.7128, -74.006], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
}

function setTraceOpen(isOpen) {
  if (!agentTrace || !traceToggle) {
    return;
  }
  agentTrace.classList.toggle("trace-list-hidden", !isOpen);
  traceToggle.setAttribute("aria-expanded", String(isOpen));
  traceToggle.textContent = isOpen ? "Hide trace" : "Show trace";
}

function clearMarkers() {
  markers.forEach((marker) => marker.remove());
  markers = [];
  routeLines.forEach((line) => line.remove());
  routeLines = [];
  distanceBadges.forEach((badge) => badge.remove());
  distanceBadges = [];
}

function updateMap(userLocation, results) {
  if (!map) {
    initMap();
  }

  clearMarkers();

  if (!userLocation || !results.length) {
    mapStatus.textContent = "The map will update after the backend finds places.";
    return;
  }

  mapStatus.textContent = `Showing ${results.length} recommendation(s) near ${userLocation.label || "your location"}.`;

  const bounds = [];
  const userMarker = L.marker([userLocation.lat, userLocation.lng], { icon: userLocationIcon }).addTo(map);
  userMarker.bindPopup(`<strong>Your location</strong><br>${userLocation.label || "Current location"}`);
  markers.push(userMarker);
  bounds.push([userLocation.lat, userLocation.lng]);

  results.forEach((place, index) => {
    if (place.lat == null || place.lng == null) {
      return;
    }

    const placeCoords = [place.lat, place.lng];
    const photoRefs = (place.photo_refs || []).slice(0, 3);
    const photoGallery = (place.photo_refs || [])
      .slice(0, 3)
      .map(
        (photoRef, photoIndex) => `
          <img
            src="/api/place-photo?name=${encodeURIComponent(photoRef)}"
            alt="${escapeHtml(`${place.name} photo ${photoIndex + 1}`)}"
            loading="lazy"
            onerror="handlePopupImageError(this)"
          >
        `
      )
      .join("");
    const hoursText = displayText(place.opening_hours_summary);
    const popup = `
      <div class="map-popup">
        ${photoGallery ? `<div class="map-popup-gallery count-${Math.min(photoRefs.length, 3)}">${photoGallery}</div>` : ""}
        <div class="map-popup-body">
          <strong class="map-popup-title">${index + 1}. ${escapeHtml(place.name)}</strong>
          ${place.address ? `<div class="map-popup-meta">${escapeHtml(place.address)}</div>` : ""}
          ${
            place.rating || place.distance_miles || place.estimated_travel_minutes
              ? `
                <div class="map-popup-stats">
                  ${place.rating ? `<span>Rating ${escapeHtml(place.rating)}</span>` : ""}
                  ${place.distance_miles ? `<span>${escapeHtml(place.distance_miles)} mi away</span>` : ""}
                  ${
                    place.estimated_travel_minutes
                      ? `<span>${escapeHtml(place.estimated_travel_minutes)} min${place.travel_mode_label ? ` ${escapeHtml(place.travel_mode_label)}` : ""}</span>`
                      : ""
                  }
                </div>
              `
              : ""
          }
          ${hoursText ? `<div class="map-popup-meta">${escapeHtml(hoursText)}</div>` : ""}
          <div class="map-popup-links">
            ${place.directions_url ? `<a href="${place.directions_url}" target="_blank" rel="noreferrer">Directions</a>` : ""}
            ${place.google_maps_url ? `<a href="${place.google_maps_url}" target="_blank" rel="noreferrer">Google Maps</a>` : ""}
          </div>
        </div>
      </div>
    `;
    const marker = L.marker(placeCoords).addTo(map);
    marker.bindPopup(popup, { maxWidth: 280, className: "restaurant-map-popup" });
    markers.push(marker);
    bounds.push(placeCoords);

    const routeLine = L.polyline(
      [
        [userLocation.lat, userLocation.lng],
        placeCoords,
      ],
      {
        color: "#4f83ff",
        weight: 3,
        opacity: 0.6,
        dashArray: "8 10",
      }
    ).addTo(map);
    routeLines.push(routeLine);

    if (place.distance_miles != null) {
      const midpoint = [
        (userLocation.lat + place.lat) / 2,
        (userLocation.lng + place.lng) / 2,
      ];
      const badge = L.marker(midpoint, {
        icon: L.divIcon({
          className: "distance-badge-wrapper",
          html: `<div class="distance-badge">${place.distance_miles} mi</div>`,
          iconSize: [64, 24],
          iconAnchor: [32, 12],
        }),
      }).addTo(map);
      distanceBadges.push(badge);
    }
  });

  if (bounds.length > 1) {
    map.fitBounds(bounds, { padding: [40, 40] });
  } else {
    map.setView(bounds[0], 14);
  }
}

async function getBrowserLocation() {
  if (!navigator.geolocation) {
    return null;
  }
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        });
      },
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 7000 }
    );
  });
}

async function createSession() {
  const data = await requestJson("/api/session", { method: "POST" });
  sessionId = data.session_id;
  await refreshSessions();
  renderMessages(data.messages || []);
  clearMarkers();
  renderAgentTrace([]);
  setTraceOpen(false);
  mapStatus.textContent = "The map will update after the backend finds places.";
}

async function refreshSessions() {
  const data = await requestJson("/api/sessions");
  sessions = data.sessions || [];
  renderSessionList();
}

async function loadSession(targetSessionId) {
  const data = await requestJson(`/api/session/${targetSessionId}`);
  sessionId = targetSessionId;
  renderMessages(data.messages || []);
  updateMap(data.user_location, data.results || []);
  renderAgentTrace(data.agent_trace || []);
  await refreshSessions();
  setDrawerOpen(false);
}

async function sendMessage(message) {
  setLoadingState(true);
  try {
    const browserLocation = await getBrowserLocation();
    return await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        browser_location: browserLocation,
      }),
    });
  } finally {
    setLoadingState(false);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  chatInput.value = "";
  addMessage("user", message);

  try {
    const data = await sendMessage(message);
    renderMessages(data.messages || []);
    updateMap(data.user_location, data.results || []);
    renderAgentTrace(data.agent_trace || []);
    await refreshSessions();
  } catch (error) {
    addMessage("assistant", `Something went wrong: ${error.message}`);
  }
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!chatInput.disabled) {
      chatForm.requestSubmit();
    }
  }
});

newChatButton.addEventListener("click", async () => {
  await createSession();
  setDrawerOpen(false);
});

chatDrawerToggle.addEventListener("click", () => {
  const isOpen = sessionSidebar.classList.contains("open");
  setDrawerOpen(!isOpen);
});

chatDrawerBackdrop.addEventListener("click", () => {
  setDrawerOpen(false);
});

chatDrawerClose.addEventListener("click", () => {
  setDrawerOpen(false);
});

traceToggle?.addEventListener("click", () => {
  const isOpen = traceToggle.getAttribute("aria-expanded") === "true";
  setTraceOpen(!isOpen);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setDrawerOpen(false);
  }
});

window.addEventListener("load", async () => {
  initMap();
  setTraceOpen(false);
  await refreshSessions();
  if (sessions.length) {
    await loadSession(sessions[0].session_id);
  } else {
    await createSession();
  }
});
