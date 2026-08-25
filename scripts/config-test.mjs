import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../config.js", import.meta.url), "utf8");
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
assert.equal(storedOverride, "https://saved.example.com");

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

assert.equal(resolve({ configuredValue: "javascript:alert(1)", protocol: "file:", origin: "null" }), "http://127.0.0.1:8787");

context.window.location.hash = "#token=session-secret";
assert.equal(context.window.resolveWearwellApiToken(), "session-secret");
assert.equal(stored.get("wearwell-api-token"), "session-secret");

console.log("Config resolution test passed");
