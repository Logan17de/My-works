/* Offline DOM, GIS, and challenge-cookie doubles. No provider/network/browser
   is contacted; these are state checks, not real Google verification claims. */
"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => {
  const watchdog = setTimeout(() => { process.stderr.write("Offline identity harness did not settle\n"); process.exitCode = 1; }, 10000);
  main(JSON.parse(input)).catch((error) => {
    process.stderr.write(error.stack + "\n"); process.exitCode = 1;
  }).finally(() => clearTimeout(watchdog));
});
class Element {
  constructor(id) { this.id = id; this.disabled = false; this.hidden = false; this.children = []; this.text = ""; this.listeners = new Map(); }
  get textContent() { return this.text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this.text = String(value); this.children = []; }
  addEventListener(name, fn) { if (!this.listeners.has(name)) this.listeners.set(name, []); this.listeners.get(name).push(fn); }
  remove() { this.removed = true; }
  async emit(name, event = {}) { await Promise.all((this.listeners.get(name) || []).map((fn) => fn(event))); }
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
const flush = async () => { await tick(); await tick(); await tick(); };
const clone = (value) => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; };
const reply = (value, status = 200) => ({ok: status >= 200 && status < 300, status, json: async () => clone(value)});
const credential = "FICTIONAL_GIS_IDTOKEN_CANARY";
const verified = {verified: true, subject: "fictional-immutable-subject", authority: "none"};
const challenge = (index = 1) => ({client_id: "12345-fictional.apps.googleusercontent.com", nonce: "N" + String(index).padStart(42, "0"), csrf: "C" + String(index).padStart(42, "0")});
function setup(fixture, options = {}) {
  const elements = new Map();
  for (const match of fs.readFileSync(fixture.assets + "/identity.html", "utf8").matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], new Element(match[1]));
  const e = (id) => elements.get(id);
  e("proof-result").hidden = true; e("google-button").hidden = true;
  const calls = [], scripts = [], configs = [], renders = [], timers = new Map(), globals = new Map();
  let index = 0, timerId = 0, activeChallenge = null, handler = null, context;
  const gis = {accounts: {id: {
    initialize: (config) => configs.push(config), renderButton: (element, config) => renders.push({element, config}),
    cancel: () => {}, disableAutoSelect: () => {},
  }}};
  const defaultHandler = async (url, request) => {
    if (url === "/identity/bootstrap") { const value = challenge(++index); activeChallenge = value; return reply(value); }
    if (url === "/identity/proof") {
      if (!activeChallenge || request.headers["X-Proof-CSRF"] !== activeChallenge.csrf) return reply({error: "csrf_mismatch", authority: "none"}, 401);
      activeChallenge = null; return reply(verified);
    }
    if (url === "/identity/cancel") {
      if (!activeChallenge || request.headers["X-Proof-CSRF"] !== activeChallenge.csrf) return reply({error: "already_consumed", authority: "none"}, 401);
      activeChallenge = null; return reply({cancelled: true, authority: "none"});
    }
    throw new Error("Unexpected UI endpoint " + url);
  };
  const loadScript = (script = scripts.at(-1)) => { context.google = gis; script.onload(); };
  const document = {documentElement: {dataset: {origin: fixture.origin}}, getElementById: (id) => elements.get(id), createElement: (tag) => new Element(tag),
    head: {appendChild: (script) => { scripts.push(script); if (options.autoScript !== false) Promise.resolve().then(() => loadScript(script)); }}};
  context = vm.createContext({document, location: {origin: options.origin || fixture.origin}, Set, Date, Error, TypeError, Object, Number, JSON,
    setTimeout: (fn, delay) => { const id = ++timerId; timers.set(id, {fn, delay}); return id; }, clearTimeout: (id) => timers.delete(id),
    addEventListener: (name, fn) => { if (!globals.has(name)) globals.set(name, []); globals.get(name).push(fn); },
    fetch: (url, request) => { assert.ok(["/identity/bootstrap", "/identity/proof", "/identity/cancel"].includes(url)); calls.push({url, options: {...request, headers: {...request.headers}}}); return handler ? handler(url, request, defaultHandler) : defaultHandler(url, request); },
  });
  vm.runInContext(fs.readFileSync(fixture.assets + "/identity.js", "utf8"), context, {filename: "identity.js"});
  return {e, calls, scripts, configs, renders, timers, context, loadScript,
    setHandler: (fn) => { handler = fn; }, setChallenge: (value) => { activeChallenge = value; },
    start: async () => { await e("start-proof").emit("click"); await flush(); },
    callback: () => configs.at(-1).callback,
    submit: async function(value = credential) { await this.callback()({credential: value}); await flush(); },
    emitGlobal: async (name, data = {}) => { for (const fn of globals.get(name) || []) await fn(data); await flush(); },
  };
}
const posts = (s, endpoint = "/identity/proof") => s.calls.filter((call) => call.url === endpoint);
function assertRequests(s) {
  for (const call of s.calls) {
    assert.equal(call.options.credentials, "same-origin"); assert.equal(call.options.cache, "no-store"); assert.equal(call.options.redirect, "error");
    assert.equal(call.options.headers.Authorization, undefined); assert.equal(call.options.headers.Cookie, undefined);
    assert.equal(call.options.headers["X-Review-CSRF"], undefined); assert.equal(call.options.headers["X-Login-CSRF"], undefined);
    assert.equal(call.options.signal, undefined, "a cookie-changing request was aborted");
    assert.ok(!call.url.includes(credential));
    if (call.url === "/identity/proof") { assert.deepEqual(JSON.parse(call.options.body), {credential}); assert.match(call.options.headers["X-Proof-CSRF"], /^[A-Za-z0-9_-]{43}$/); }
    else assert.ok(!String(call.options.body).includes(credential));
  }
}
async function main(fixture) {
  const completed = [];
  let s = setup(fixture); await flush();
  assert.equal(s.calls.length, 0); assert.equal(s.scripts.length, 0); assert.equal(s.e("start-proof").disabled, false); assert.equal(s.e("cancel-proof").disabled, true);
  await s.start();
  assert.equal(s.scripts.length, 1); assert.equal(s.scripts[0].src, "https://accounts.google.com/gsi/client");
  assert.equal(s.configs[0].client_id, challenge().client_id); assert.equal(s.configs[0].nonce, challenge().nonce); assert.equal(s.configs[0].auto_select, false);
  assert.equal(s.e("google-button").hidden, false);
  await s.submit(); await s.callback()({credential});
  assert.equal(posts(s).length, 1); assert.equal(posts(s)[0].options.headers["X-Proof-CSRF"], challenge().csrf);
  assert.equal(s.e("proof-result").hidden, false); assert.equal(s.e("proof-subject").textContent, verified.subject);
  assert.equal(s.e("proof-authority").textContent, "Authority: none"); assert.equal(s.e("proof-subject").children.length, 0);
  await s.e("cancel-proof").emit("click"); assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("proof-subject").textContent, "");
  assert.equal(posts(s, "/identity/cancel").length, 0, "clearing a consumed displayed result made an unnecessary request");
  assertRequests(s); completed.push("explicit start, pinned GIS/nonce/proof-CSRF, one proof POST, verified text-only subject and authority none");

  s = setup(fixture); const boot = deferred();
  s.setHandler(async (url, request, fallback) => url === "/identity/bootstrap" ? boot.promise : fallback(url, request));
  const booting = s.e("start-proof").emit("click"); await flush(); await s.e("start-proof").emit("click");
  const cancelBootstrap = s.e("cancel-proof").emit("click"); await flush();
  assert.equal(s.e("start-proof").disabled, true); assert.equal(s.calls.length, 1); assert.equal(s.scripts.length, 0);
  s.setChallenge(challenge()); boot.resolve(reply(challenge())); await booting; await cancelBootstrap;
  assert.equal(s.scripts.length, 0); assert.equal(s.configs.length, 0); assert.equal(s.e("start-proof").disabled, false);
  assert.equal(posts(s, "/identity/cancel")[0].options.headers["X-Proof-CSRF"], challenge().csrf);
  assertRequests(s); completed.push("repeat start and Cancel during bootstrap serialize the late challenge cookie before cleanup");

  s = setup(fixture, {autoScript: false}); const loading = s.e("start-proof").emit("click"); await flush();
  await s.e("cancel-proof").emit("click"); assert.equal(s.e("start-proof").disabled, false);
  s.loadScript(); await loading; assert.equal(s.configs.length, 0); assert.equal(s.e("google-button").hidden, true);
  await s.start(); assert.equal(s.configs.length, 1); assert.equal(s.configs[0].nonce, challenge(2).nonce); assert.equal(s.scripts.length, 1);
  completed.push("Cancel during GIS load prevents stale rendering and permits a fresh nonce");

  s = setup(fixture); await s.start(); const oldCallback = s.callback();
  await s.start(); assert.equal(s.configs.length, 2); assert.equal(s.configs[1].nonce, challenge(2).nonce);
  await oldCallback({credential}); assert.equal(posts(s).length, 0);
  await s.submit(); assert.equal(posts(s).length, 1);
  await oldCallback({credential}); assert.equal(posts(s).length, 1);
  completed.push("restart fences old GIS callbacks and binds the new proof to its fresh challenge");

  s = setup(fixture); await s.start(); const proof = deferred();
  s.setHandler(async (url, request, fallback) => url === "/identity/proof" ? proof.promise : fallback(url, request));
  const submitting = s.callback()({credential}); await flush(); await s.callback()({credential}); await s.e("start-proof").emit("click");
  assert.equal(posts(s).length, 1); assert.equal(posts(s, "/identity/bootstrap").length, 1);
  const cancelling = s.e("cancel-proof").emit("click"); await flush();
  assert.equal(s.e("proof-result").hidden, true); assert.ok(s.e("proof-status").textContent.includes("still in flight"));
  await s.e("start-proof").emit("click"); assert.equal(posts(s, "/identity/bootstrap").length, 1);
  s.setChallenge(null); proof.resolve(reply(verified)); await submitting; await cancelling;
  assert.equal(s.e("proof-subject").textContent, ""); assert.equal(s.e("start-proof").disabled, false);
  assertRequests(s); completed.push("duplicate proof callback, restart blocked during POST, Cancel waits and discards the late verified result");

  for (const failure of [reply({error: credential, authority: "none"}, 401), {ok: true, status: 200, json: async () => { throw new Error(credential); }}]) {
    s = setup(fixture); await s.start(); s.setHandler(async (url, request, fallback) => url === "/identity/proof" ? failure : fallback(url, request));
    await s.submit(); await s.callback()({credential});
    assert.equal(posts(s).length, 1); assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("proof-subject").textContent, "");
    assert.ok(!s.e("proof-status").textContent.includes("CANARY")); assert.ok(s.e("proof-status").textContent.includes("will not be resubmitted"));
  }
  completed.push("refused/mismatched and malformed proof responses are redacted and never automatically retried");

  s = setup(fixture); await s.start(); s.setHandler(async (url, request, fallback) => { if (url === "/identity/proof") throw new TypeError(credential); return fallback(url, request); });
  await s.submit(); await s.callback()({credential}); await flush();
  assert.equal(posts(s).length, 1); assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("start-proof").disabled, false);
  assert.ok(!s.e("proof-status").textContent.includes("CANARY"));
  completed.push("uncertain proof POST remains unconfirmed with an explicit fresh-start option and no retry");

  const invalidProofs = [{...verified, authority: "reviewer"}, {...verified, verified: false}, {...verified, subject: credential}, {...verified, subject: "person@example.invalid"}, {...verified, subject: "space subject"}, {...verified, subject: "é"}, {...verified, subject: "x".repeat(256)}, {...verified, secret: credential}];
  for (const value of invalidProofs) {
    s = setup(fixture); await s.start(); s.setHandler(async (url, request, fallback) => url === "/identity/proof" ? reply(value) : fallback(url, request));
    await s.submit(); assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("proof-subject").textContent, ""); assert.ok(!s.e("proof-status").textContent.includes("CANARY"));
  }
  completed.push("unexpected authority, false verification, credential reflection, invalid subject, and extra fields fail closed");

  s = setup(fixture); await s.start();
  s.setHandler(async (url, request, fallback) => url === "/identity/proof" ? reply({...verified, subject: "<script>FICTIONAL_CANARY</script>"}) : fallback(url, request));
  await s.submit(); assert.equal(s.e("proof-subject").textContent, "<script>FICTIONAL_CANARY</script>"); assert.equal(s.e("proof-subject").children.length, 0);
  completed.push("a printable subject containing markup stays literal text without created DOM elements");

  for (const value of [{...challenge(), client_id: "https://evil.example.invalid"}, {...challenge(), nonce: "short"}, {...challenge(), csrf: "short"}]) {
    s = setup(fixture); s.setHandler(async (url, request, fallback) => url === "/identity/bootstrap" ? reply(value) : fallback(url, request));
    await s.start(); assert.equal(s.scripts.length, 0); assert.equal(s.e("google-button").hidden, true); assert.equal(posts(s).length, 0);
  }
  completed.push("invalid public client ID, nonce, or proof-CSRF rejected before loading GIS");

  for (const value of ["", "white space", "x".repeat(8193)]) {
    s = setup(fixture); await s.start(); await s.submit(value); await s.callback()({credential});
    assert.equal(posts(s).length, 0); assert.equal(s.e("proof-result").hidden, true);
  }
  completed.push("invalid or oversized credential is neither submitted nor reflected");

  s = setup(fixture); await s.start(); const replayed = s.callback(); await s.submit(); await s.start();
  await replayed({credential}); assert.equal(posts(s).length, 1);
  s.setHandler(async (url, request, fallback) => url === "/identity/proof" ? reply({error: "fictional_nonce_mismatch", authority: "none"}, 401) : fallback(url, request));
  await s.submit(); assert.equal(posts(s).length, 2); assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("proof-subject").textContent, "");
  completed.push("replayed old callback is fenced while a server-refused fresh proof never displays a subject");

  s = setup(fixture); await s.start(); await s.submit(); await s.start();
  assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("proof-subject").textContent, "");
  await s.submit(); await s.emitGlobal("pagehide"); await s.emitGlobal("pageshow", {persisted: true});
  assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("proof-subject").textContent, ""); assert.equal(s.calls.length, 4);
  completed.push("new proof and Back/Forward clear the result without automatic bootstrap or verification");

  s = setup(fixture); await s.start(); const historyProof = deferred();
  s.setHandler(async (url, request, fallback) => url === "/identity/proof" ? historyProof.promise : fallback(url, request));
  const navigating = s.callback()({credential}); await flush(); await s.emitGlobal("pagehide"); await s.emitGlobal("pageshow", {persisted: true});
  await s.e("start-proof").emit("click"); assert.equal(posts(s, "/identity/bootstrap").length, 1);
  historyProof.resolve(reply(verified)); await navigating; await flush();
  assert.equal(s.e("proof-result").hidden, true); assert.equal(s.e("start-proof").disabled, false);
  completed.push("history during proof waits for settlement and ignores the stale result before enabling a new start");

  s = setup(fixture); const historyBoot = deferred();
  s.setHandler(async (url, request, fallback) => url === "/identity/bootstrap" ? historyBoot.promise : fallback(url, request));
  const oldBooting = s.e("start-proof").emit("click"); await flush(); await s.emitGlobal("pagehide"); await s.emitGlobal("pageshow", {persisted: true});
  await s.e("start-proof").emit("click"); assert.equal(s.calls.length, 1);
  historyBoot.resolve(reply(challenge())); await oldBooting;
  assert.equal(s.scripts.length, 0); assert.equal(s.e("start-proof").disabled, false);
  completed.push("history during bootstrap does not abort or race the old challenge cookie");

  s = setup(fixture); await s.start(); const cancelDelay = deferred();
  s.setHandler(async (url, request, fallback) => url === "/identity/cancel" ? cancelDelay.promise : fallback(url, request));
  const pendingCancel = s.e("cancel-proof").emit("click"); await flush(); const repeatedCancel = s.e("cancel-proof").emit("click");
  await s.e("start-proof").emit("click"); assert.equal(posts(s, "/identity/cancel").length, 1); assert.equal(posts(s, "/identity/bootstrap").length, 1);
  cancelDelay.resolve(reply({cancelled: true, authority: "none"})); await pendingCancel; await repeatedCancel;
  assert.equal(s.e("start-proof").disabled, false); completed.push("duplicate Cancel and restart while cleanup is pending remain serialized");

  s = setup(fixture); await s.start();
  s.setHandler(async (url, request, fallback) => { if (url === "/identity/cancel") throw new TypeError(credential); return fallback(url, request); });
  await s.e("cancel-proof").emit("click"); assert.equal(posts(s, "/identity/cancel").length, 1); assert.equal(s.e("proof-result").hidden, true);
  assert.ok(s.e("proof-status").textContent.includes("could not be confirmed")); assert.ok(!s.e("proof-status").textContent.includes("CANARY"));
  await flush(); assert.equal(posts(s, "/identity/cancel").length, 1);
  completed.push("uncertain challenge cancellation is redacted without an automatic retry");

  s = setup(fixture, {autoScript: false}); const timeoutStart = s.e("start-proof").emit("click"); await flush();
  const timer = Array.from(s.timers.values())[0]; assert.equal(timer.delay, 15000); timer.fn(); await timeoutStart;
  assert.equal(s.scripts[0].removed, true); s.loadScript(s.scripts[0]); assert.equal(s.configs.length, 0);
  assert.equal(s.e("start-proof").disabled, false); assert.equal(s.e("cancel-proof").disabled, false);
  completed.push("GIS timeout removes the script, ignores late load, and leaves the challenge cancellable");

  s = setup(fixture, {origin: "https://evil.example.invalid"}); await s.e("start-proof").emit("click"); await s.e("cancel-proof").emit("click"); await s.emitGlobal("pageshow", {persisted: true});
  assert.equal(s.calls.length, 0); assert.equal(s.scripts.length, 0); assert.equal(s.e("start-proof").disabled, true);
  completed.push("foreign-origin start, Cancel, and history cannot contact proof or Google endpoints");

  process.stdout.write(JSON.stringify({result: "identity socket-free state checks passed", scenarios: completed}, null, 2) + "\n");
}
