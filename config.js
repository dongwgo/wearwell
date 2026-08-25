(() => {
  const existing = window.WEARWELL_CONFIG || {};
  window.WEARWELL_CONFIG = { API_BASE: "", API_TOKEN: "", ...existing };

  const safeStorageGet = key => {
    try { return window.localStorage?.getItem(key) || ""; } catch { return ""; }
  };
  const safeStorageSet = (key, value) => {
    try { window.localStorage?.setItem(key, value); } catch {}
  };
  const cleanHttpUrl = value => {
    try {
      const url = new URL(String(value || "").trim());
      return ["http:", "https:"].includes(url.protocol) ? url.href.replace(/\/+$/, "") : "";
    } catch { return ""; }
  };

  window.resolveWearwellApiBase = ({
    storageValue = safeStorageGet("오늘옷-api"),
    configuredValue = window.WEARWELL_CONFIG.API_BASE,
    protocol = window.location.protocol,
    origin = window.location.origin
  } = {}) => {
    const explicit = cleanHttpUrl(storageValue) || cleanHttpUrl(configuredValue);
    if (explicit) return explicit;
    return protocol === "http:" || protocol === "https:" ? cleanHttpUrl(origin) : "http://127.0.0.1:8787";
  };

  window.resolveWearwellApiToken = () => {
    const hash = String(window.location.hash || "");
    const match = hash.match(/(?:^#|&)token=([A-Za-z0-9_-]{12,128})(?:&|$)/);
    if (match) {
      safeStorageSet("wearwell-api-token", match[1]);
      try { window.history.replaceState(null, "", window.location.pathname + window.location.search); } catch {}
      return match[1];
    }
    return String(window.WEARWELL_CONFIG.API_TOKEN || safeStorageGet("wearwell-api-token") || "");
  };
})();
