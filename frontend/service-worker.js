/* ============================================================
   Virtual Closet — Service Worker
   Cache-first for static assets, network-first for API calls
   ============================================================ */

const CACHE_NAME  = "virtual-closet-v1";
const CACHE_URLS  = [
  "./index.html",
  "./style.css",
  "./script.js",
  "./manifest.json",
  "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
];

/* ── Install: pre-cache shell ── */
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(CACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

/* ── Activate: clear old caches ── */
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

/* ── Fetch strategy ── */
self.addEventListener("fetch", event => {
  const { request } = event;
  const url = new URL(request.url);

  /* Pass through non-GET and backend API calls */
  if (request.method !== "GET") return;
  if (url.hostname === "127.0.0.1" || url.hostname === "localhost") return;

  /* Cache-first for same-origin static assets */
  event.respondWith(
    caches.match(request).then(cached => {
      if (cached) return cached;

      return fetch(request).then(response => {
        /* Only cache valid, same-origin or font responses */
        if (
          response.ok &&
          (url.origin === self.location.origin || url.hostname.includes("fonts.g"))
        ) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
        }
        return response;
      }).catch(() => {
        /* Offline fallback for navigation requests */
        if (request.destination === "document") {
          return caches.match("./index.html");
        }
      });
    })
  );
});
