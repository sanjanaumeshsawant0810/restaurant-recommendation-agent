const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const newChatButton = document.getElementById("new-chat-button");
const chatDrawerToggle = document.getElementById("chat-drawer-toggle");
const chatDrawerBackdrop = document.getElementById("chat-drawer-backdrop");
const chatDrawerClose = document.getElementById("chat-drawer-close");
const sessionSidebar = document.getElementById("session-sidebar");
const mapStatus = document.getElementById("map-status");
const agentTrace = document.getElementById("agent-trace");
const projectSummary = document.getElementById("project-summary");
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

function setDrawerOpen(isOpen) {
  sessionSidebar.classList.toggle("open", isOpen);
  chatDrawerBackdrop.classList.toggle("open", isOpen);
  chatDrawerToggle.setAttribute("aria-expanded", String(isOpen));
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.message || `Request failed with status ${response.status}`);
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
}

function renderMessages(messages = []) {
  chatLog.innerHTML = "";
  messages.forEach((message) => addMessage(message.role, message.text));
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

function renderProjectSummary(summary) {
  projectSummary.innerHTML = "";
  const architecture = summary?.architecture || [];
  if (!architecture.length) {
    return;
  }
  architecture.forEach((item) => {
    const node = document.createElement("li");
    node.textContent = item;
    projectSummary.appendChild(node);
  });
}

function initMap() {
  map = L.map("map").setView([40.7128, -74.006], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
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
    const popup = `
      <strong>${index + 1}. ${place.name}</strong><br>
      ${place.address || ""}<br>
      ${place.rating ? `Rating: ${place.rating}<br>` : ""}
      ${place.distance_miles ? `Distance: ${place.distance_miles} miles<br>` : ""}
      ${place.estimated_travel_minutes ? `Travel time: ${place.estimated_travel_minutes} min<br>` : ""}
      ${place.directions_url ? `<a href="${place.directions_url}" target="_blank" rel="noreferrer">Get directions</a><br>` : ""}
      ${place.google_maps_url ? `<a href="${place.google_maps_url}" target="_blank" rel="noreferrer">Open in Google Maps</a>` : ""}
    `;
    const marker = L.marker(placeCoords).addTo(map);
    marker.bindPopup(popup);
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
  renderProjectSummary({});
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
  renderProjectSummary(data.project_summary || {});
  await refreshSessions();
  setDrawerOpen(false);
}

async function sendMessage(message) {
  sendButton.disabled = true;
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
    sendButton.disabled = false;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  chatInput.value = "";

  try {
    const data = await sendMessage(message);
    renderMessages(data.messages || []);
    updateMap(data.user_location, data.results || []);
    renderAgentTrace(data.agent_trace || []);
    renderProjectSummary(data.project_summary || {});
    await refreshSessions();
  } catch (error) {
    addMessage("assistant", `Something went wrong: ${error.message}`);
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

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setDrawerOpen(false);
  }
});

window.addEventListener("load", async () => {
  initMap();
  await refreshSessions();
  if (sessions.length) {
    await loadSession(sessions[0].session_id);
  } else {
    await createSession();
  }
});
