import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../config.js", import.meta.url), "utf8");
const index = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const stored = new Map();
const context = {
  URL,
  window: {
    localStorage: {
      getItem: key => stored.get(key) || null,
      setItem: (key, value) => stored.set(key, value)
    },
    location: { protocol: "https:", origin: "https://frontend.example.com", hash: "" },
    history: { replaceState() {} }
  }
};
vm.createContext(context);
vm.runInContext(source, context);

const resolve = context.window.resolveWearwellApiBase;
assert.equal(typeof resolve, "function");

const configured = resolve({
  storageValue: "",
  queryValue: "",
  configuredValue: "https://colab.example.com/",
  protocol: "https:",
  origin: "https://frontend.example.com"
});
assert.equal(configured, "https://colab.example.com");

const maliciousQueryIgnored = resolve({
  storageValue: "",
  configuredValue: "https://configured.example.com",
  protocol: "https:",
  origin: "https://frontend.example.com"
});
assert.equal(maliciousQueryIgnored, "https://configured.example.com");

const storedOverride = resolve({
  storageValue: "https://saved.example.com/",
  queryValue: "https://query.example.com",
  configuredValue: "https://configured.example.com",
  protocol: "file:",
  origin: "null"
});
assert.equal(storedOverride, "https://configured.example.com");

const invalidConfigured = resolve({
  storageValue: "https://stale-attacker.example.com",
  configuredValue: "javascript:alert(1)",
  protocol: "http:",
  origin: "http://127.0.0.1:8000"
});
assert.equal(invalidConfigured, "");

const sameOrigin = resolve({
  storageValue: "",
  queryValue: "",
  configuredValue: "",
  protocol: "https:",
  origin: "https://wearwell.trycloudflare.com"
});
assert.equal(sameOrigin, "https://wearwell.trycloudflare.com");

const localFallback = resolve({
  storageValue: "",
  queryValue: "",
  configuredValue: "",
  protocol: "file:",
  origin: "null"
});
assert.equal(localFallback, "http://127.0.0.1:8787");

assert.equal(resolve({ configuredValue: "javascript:alert(1)", protocol: "file:", origin: "null" }), "");

context.window.WEARWELL_CONFIG.API_TOKEN = "local-file-secret";
context.window.WEARWELL_CONFIG.API_BASE = "javascript:alert(1)";
context.window.location.hash = "#token=url-secret-value";
assert.equal(context.window.resolveWearwellApiToken(), "");
assert.equal(stored.has("wearwell-api-token"), false);

const localConfigPosition = index.indexOf('src="local-config.js"');
assert.ok(localConfigPosition >= 0);
assert.ok(localConfigPosition < index.indexOf('src="config.js"'));

console.log("Config resolution test passed");
