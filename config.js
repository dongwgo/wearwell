(() => {
  const existing = window.WEARWELL_CONFIG || {};
  window.WEARWELL_CONFIG = { API_BASE: "", API_TOKEN: "", LOCAL_API_BASE: "http://127.0.0.1:8787", LOCAL_API_TOKEN: "", ...existing };

  const safeStorageGet = key => {
    try { return window.localStorage?.getItem(key) || ""; } catch { return ""; }
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
    const configuredInput = String(configuredValue || "").trim();
    if (configuredInput) return cleanHttpUrl(configuredInput);
    const stored = cleanHttpUrl(storageValue);
    if (stored) return stored;
    return protocol === "http:" || protocol === "https:" ? cleanHttpUrl(origin) : "http://127.0.0.1:8787";
  };

  window.resolveWearwellApiToken = () => {
    const configuredToken = String(window.WEARWELL_CONFIG.API_TOKEN || "");
    if (configuredToken) return cleanHttpUrl(window.WEARWELL_CONFIG.API_BASE) ? configuredToken : "";
    return String(safeStorageGet("wearwell-api-token") || "");
  };
})();
