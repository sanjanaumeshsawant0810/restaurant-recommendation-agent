const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const mapStatus = document.getElementById("map-status");
const agentTrace = document.getElementById("agent-trace");
const projectSummary = document.getElementById("project-summary");

let sessionId = null;
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

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
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
  const response = await fetch("/api/session", { method: "POST" });
  const data = await response.json();
  sessionId = data.session_id;
}

async function sendMessage(message) {
  sendButton.disabled = true;

  const browserLocation = await getBrowserLocation();
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      browser_location: browserLocation,
    }),
  });

  const data = await response.json();
  sendButton.disabled = false;
  return data;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  chatInput.value = "";

  try {
    const data = await sendMessage(message);
    addMessage("assistant", data.reply || "I hit an error.");
    updateMap(data.user_location, data.results || []);
    renderAgentTrace(data.agent_trace || []);
    renderProjectSummary(data.project_summary || {});
  } catch (error) {
    addMessage("assistant", `Something went wrong: ${error.message}`);
  }
});

window.addEventListener("load", async () => {
  initMap();
  await createSession();
  addMessage(
    "assistant",
    "Tell me what you want to eat. I’ll ask for anything missing, then I’ll show the top 5 by default. Ask for top ten if you want more."
  );
  renderAgentTrace([]);
});
