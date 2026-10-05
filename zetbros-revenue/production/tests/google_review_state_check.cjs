/* Offline DOM/GIS/fetch doubles only. This does not load Google, use real
   credentials, or establish browser rendering or provider compatibility. */
"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => main(JSON.parse(input)).catch((error) => {
  process.stderr.write(error.stack + "\n"); process.exitCode = 1;
}));

class Element {
  constructor(id) {
    this.id = id; this.value = ""; this.disabled = false; this.hidden = false;
    this.checked = false; this.listeners = new Map(); this.children = []; this.text = "";
  }
  get textContent() { return this.text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this.text = String(value); this.children = []; }
  addEventListener(name, fn) { if (!this.listeners.has(name)) this.listeners.set(name, []); this.listeners.get(name).push(fn); }
  append(...children) { this.children.push(...children); }
  remove() { this.removed = true; }
  async emit(name, event = {}) {
    event.preventDefault = () => {};
    await Promise.all((this.listeners.get(name) || []).map((fn) => fn(event)));
  }
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
const flush = async () => { await tick(); await tick(); await tick(); };
const clone = (value) => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; };
const reply = (value, status = 200) => ({ok: status >= 200 && status < 300, status, json: async () => clone(value)});
const credential = "FICTIONAL_GIS_ID_TOKEN_CANARY";
const csrf = "fictional_session_csrf_0123456789";
const bootstrap = {mode: "google_oidc", client_id: "fictional-client.apps.googleusercontent.com", nonce: "fictional_nonce_0123456789", csrf: "fictional_login_csrf_0123456789"};

function setup(fixture, options = {}) {
  const elements = new Map();
  for (const match of fs.readFileSync(fixture.assets + "/review.html", "utf8").matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], new Element(match[1]));
  for (const id of ["preview", "wire-section", "google-session", "google-button", "google-retry"]) elements.get(id).hidden = true;
  if (options.removeBearer) for (const id of ["session-form", "reviewer-token", "unlock"]) elements.delete(id);
  const e = (id) => elements.get(id);
  const scripts = [], calls = [], initializations = [], renders = [], timers = new Map(), globals = new Map();
  let timerId = 0, session = options.restored === true, sessionCsrf = csrf;
  let handler = options.handler || null, context;
  const gis = {accounts: {id: {
    initialize: (config) => initializations.push(config),
    renderButton: (element, config) => renders.push({element, config}),
    cancel: () => {}, disableAutoSelect: () => {},
  }}};
  const defaultHandler = async (url, request) => {
    if (url === "/review/auth/session") return session ? reply({authenticated: true, csrf: sessionCsrf}) : reply({error: "authentication_required"}, 401);
    if (url === "/review/auth/bootstrap") return reply(bootstrap);
    if (url === "/review/auth/google") { session = true; return reply({authenticated: true, csrf: sessionCsrf}); }
    if (url === "/review/auth/logout") { session = false; return reply({authenticated: false}); }
    if (url.endsWith("/decision")) return reply({...fixture.proposal, state: JSON.parse(request.body).decision === "approve" ? "approved" : "denied"});
    return reply(fixture.proposal);
  };
  const loadScript = (script = scripts.at(-1)) => { context.google = gis; script.onload(); };
  const document = {
    documentElement: {dataset: {reviewOrigin: fixture.origin, authMode: options.mode || "google_oidc", workflowState: options.workflowState || "configured"}},
    getElementById: (id) => elements.get(id), createElement: (tag) => new Element(tag),
    head: {appendChild: (script) => { scripts.push(script); if (options.autoScript !== false) Promise.resolve().then(() => loadScript(script)); }},
  };
  context = vm.createContext({document, location: {origin: options.origin || fixture.origin}, AbortController, Set, Date, Error, TypeError, Object, Number, JSON,
    setTimeout: (fn) => { const id = ++timerId; timers.set(id, fn); return id; }, clearTimeout: (id) => timers.delete(id),
    addEventListener: (event, fn) => { if (!globals.has(event)) globals.set(event, []); globals.get(event).push(fn); },
    fetch: (url, request) => {
      assert.ok(url.startsWith("/review/"), "fetch left the relative trusted review surface");
      calls.push({url, options: {...request, headers: {...request.headers}}});
      return handler ? handler(url, request, defaultHandler) : defaultHandler(url, request);
    },
  });
  vm.runInContext(fs.readFileSync(fixture.assets + "/review.js", "utf8"), context, {filename: "review.js"});
  const env = {e, calls, scripts, initializations, renders, timers, context, loadScript,
    setHandler: (fn) => { handler = fn; }, setSession: (active, value = csrf) => { session = active; sessionCsrf = value; },
    callback: () => initializations.at(-1).callback,
    signIn: async () => { await env.callback()({credential}); await flush(); },
    input: async (value) => { e("proposal-id").value = value; await e("proposal-id").emit("input"); },
    load: async (id = fixture.proposal.id) => { await env.input(id); await e("proposal-form").emit("submit"); },
    confirm: async () => { e("confirm-preview").checked = true; await e("confirm-preview").emit("change"); },
    emitGlobal: async (event, data = {}) => { for (const fn of globals.get(event) || []) await fn(data); await flush(); },
  };
  return env;
}
function loginPosts(s) { return s.calls.filter((call) => call.url === "/review/auth/google"); }
function decisionPosts(s) { return s.calls.filter((call) => call.url.endsWith("/decision")); }
function assertGoogleRequests(s) {
  for (const call of s.calls) {
    assert.equal(call.options.credentials, "same-origin");
    assert.equal(call.options.cache, "no-store");
    assert.equal(call.options.redirect, "error");
    assert.equal(call.options.headers.Authorization, undefined);
    assert.equal(call.options.headers.Cookie, undefined);
    assert.ok(!call.url.includes(credential));
    if (call.url !== "/review/auth/google") assert.ok(!String(call.options.body).includes(credential));
    if (call.url.endsWith("/decision") || call.url === "/review/auth/logout") assert.ok(call.options.headers["X-Review-CSRF"]);
  }
}

async function main(fixture) {
  const completed = [];
  let s = setup(fixture); await flush();
  assert.equal(s.e("session-form").hidden, true);
  assert.equal(s.e("reviewer-token").disabled, true);
  assert.equal(s.e("google-session").hidden, false);
  assert.equal(s.e("load").disabled, true);
  assert.equal(s.e("google-button").hidden, false);
  assert.equal(s.scripts.length, 1);
  assert.equal(s.scripts[0].src, "https://accounts.google.com/gsi/client");
  assert.equal(s.initializations[0].client_id, bootstrap.client_id);
  assert.equal(s.initializations[0].nonce, bootstrap.nonce);
  assert.equal(s.initializations[0].auto_select, false);
  await s.signIn();
  assert.deepEqual(JSON.parse(loginPosts(s)[0].options.body), {credential});
  assert.equal(loginPosts(s)[0].options.headers["X-Login-CSRF"], bootstrap.csrf);
  assert.equal(loginPosts(s)[0].options.headers["X-Review-CSRF"], undefined);
  assert.equal(s.e("load").disabled, false);
  assert.equal(s.e("google-button").hidden, true);
  await s.load();
  assert.equal(s.e("reply-body").textContent, fixture.proposal.payload.body);
  assert.equal(s.e("wire-body").textContent, fixture.proposal.wire_preview.body);
  assert.equal(s.e("reply-body").children.length, 0);
  assert.deepEqual(JSON.parse(s.e("action").textContent), fixture.proposal.payload);
  assert.deepEqual(JSON.parse(s.e("wire").textContent), fixture.proposal.wire_preview);
  assert.ok(s.e("identity").textContent.includes(fixture.proposal.digest));
  assert.equal(s.e("approve").disabled, true);
  await s.confirm(); await s.e("approve").emit("click"); await s.e("approve").emit("click");
  assert.equal(decisionPosts(s).length, 1);
  assert.deepEqual(JSON.parse(decisionPosts(s)[0].options.body), {digest: fixture.proposal.digest, decision: "approve"});
  assert.equal(decisionPosts(s)[0].options.headers["X-Review-CSRF"], csrf);
  assert.ok(s.e("status").textContent.includes("Decision recorded"));
  await s.e("logout").emit("click");
  assert.equal(s.e("load").disabled, true);
  assert.equal(s.e("preview").hidden, true);
  assert.equal(s.e("proposal-id").value, "");
  for (const id of ["identity", "reply-body", "action", "wire", "wire-body", "ledger"]) assert.equal(s.e(id).textContent, "");
  const logout = s.calls.find((call) => call.url === "/review/auth/logout");
  assert.equal(logout.options.headers["X-Review-CSRF"], csrf);
  assert.equal(s.initializations.length, 2);
  assert.equal(s.scripts.length, 1, "GIS script was unnecessarily loaded again");
  assertGoogleRequests(s); completed.push("GIS nonce/login-CSRF, ID-token-only login, exact preview/digest, review-CSRF, repeated click, logout");

  s = setup(fixture, {restored: true, removeBearer: true}); await flush();
  assert.equal(s.e("load").disabled, false);
  assert.equal(s.scripts.length, 0, "Google loaded although an existing server session sufficed");
  await s.load(); await s.confirm(); await s.e("deny").emit("click");
  assert.equal(decisionPosts(s)[0].options.headers["X-Review-CSRF"], csrf);
  assertGoogleRequests(s); completed.push("session restoration without GIS or bearer elements");

  s = setup(fixture); await flush(); const oldCallback = s.callback(); const login = deferred();
  s.setHandler(async (url, request, fallback) => url === "/review/auth/google" ? login.promise : fallback(url, request));
  const signingIn = oldCallback({credential}); await flush();
  await oldCallback({credential}); assert.equal(loginPosts(s).length, 1);
  const cancellingLogin = s.e("cancel").emit("click"); await flush();
  assert.equal(s.e("load").disabled, true); assert.equal(s.e("logout").disabled, true);
  assert.equal(loginPosts(s)[0].options.signal, undefined, "login POST was aborted");
  s.setSession(true, "fictional_late_login_csrf_012345");
  login.resolve(reply({authenticated: true, csrf: "fictional_late_login_csrf_012345"}));
  await signingIn; await cancellingLogin; await oldCallback({credential});
  assert.equal(s.e("load").disabled, true);
  assert.equal(loginPosts(s).length, 1);
  assert.equal(s.calls.find((call) => call.url === "/review/auth/logout").options.headers["X-Review-CSRF"], "fictional_late_login_csrf_012345");
  assertGoogleRequests(s); completed.push("duplicate callback and Cancel during login clear the late cookie session");

  s = setup(fixture, {autoScript: false}); await flush();
  assert.equal(s.scripts.length, 1);
  const cancelLibrary = s.e("cancel").emit("click"); await flush();
  assert.equal(s.initializations.length, 0);
  s.loadScript(); await cancelLibrary; await flush();
  assert.equal(s.initializations.length, 1, "stale script continuation rendered a second GIS button");
  assert.equal(s.e("load").disabled, true);
  completed.push("Cancel during GIS loading fences stale authentication continuation");

  s = setup(fixture, {restored: true}); await flush(); await s.load(); await s.confirm();
  const uncertain = deferred();
  s.setHandler(async (url, request, fallback) => url.endsWith("/decision") ? uncertain.promise : fallback(url, request));
  const decision = s.e("approve").emit("click"); await flush();
  await s.e("approve").emit("click"); await s.e("cancel").emit("click");
  assert.equal(s.e("preview").hidden, true);
  assert.ok(s.e("status").textContent.includes("still in flight"));
  assert.equal(decisionPosts(s)[0].options.signal, undefined, "decision POST was aborted");
  await s.callback()({credential}); assert.equal(loginPosts(s).length, 0, "login proceeded during decision");
  uncertain.reject(new TypeError("FICTIONAL_SECRET_ERROR_CANARY")); await decision;
  assert.ok(s.e("status").textContent.includes("has settled"));
  await s.signIn(); await s.load(); await s.confirm(); await s.e("approve").emit("click");
  assert.equal(decisionPosts(s).length, 1);
  assert.equal(s.e("approve").disabled, true);
  assert.ok(s.e("status").textContent.includes("already attempted"));
  assertGoogleRequests(s); completed.push("in-flight decision Cancel, no abort, no uncertain retry across logout/login");

  s = setup(fixture, {restored: true}); await flush();
  const oldLoad = deferred(); const newer = clone(fixture.proposal);
  newer.id = "00000000-0000-0000-0000-000000000002"; newer.payload.body = "Newer exact preview";
  s.setHandler(async (url, request, fallback) => url.endsWith(fixture.proposal.id) ? oldLoad.promise : url.endsWith(newer.id) ? reply(newer) : fallback(url, request));
  await s.input(fixture.proposal.id); const firstLoad = s.e("proposal-form").emit("submit"); await flush();
  await s.load(newer.id); oldLoad.resolve(reply(fixture.proposal)); await firstLoad;
  assert.equal(s.e("reply-body").textContent, "Newer exact preview");
  assert.equal(s.calls.find((call) => call.url.endsWith(fixture.proposal.id)).options.signal.aborted, true);
  const cancelledLoad = deferred(); s.setHandler(async (url, request, fallback) => url.endsWith(fixture.proposal.id) ? cancelledLoad.promise : fallback(url, request));
  await s.input(fixture.proposal.id); const pendingLoad = s.e("proposal-form").emit("submit"); await flush();
  await s.e("cancel").emit("click"); cancelledLoad.resolve(reply(fixture.proposal)); await pendingLoad;
  assert.equal(s.e("preview").hidden, true); assert.equal(s.e("load").disabled, true);
  completed.push("racing and cancelled preview GETs cannot revive stale preview");

  s = setup(fixture, {restored: true}); await flush();
  const staleUnauthorized = deferred();
  s.setHandler(async (url, request, fallback) => url.endsWith(fixture.proposal.id) ? staleUnauthorized.promise : url.endsWith(newer.id) ? reply(newer) : fallback(url, request));
  await s.input(fixture.proposal.id); const staleRequest = s.e("proposal-form").emit("submit"); await flush();
  await s.load(newer.id); staleUnauthorized.resolve(reply({error: credential}, 401)); await staleRequest;
  assert.equal(s.e("load").disabled, false); assert.equal(s.e("reply-body").textContent, newer.payload.body);
  await s.confirm(); assert.equal(s.e("approve").disabled, false);
  completed.push("stale unauthorized GET cannot clear a newer verified session or preview");

  for (const bad of [{...bootstrap, mode: "bearer"}, {...bootstrap, client_id: "https://evil.example.invalid/client"}, {...bootstrap, nonce: "short"}, {...bootstrap, csrf: "short"}]) {
    s = setup(fixture); s.setHandler(async (url, request, fallback) => url === "/review/auth/bootstrap" ? reply(bad) : fallback(url, request)); await flush();
    assert.equal(s.scripts.length, 0); assert.equal(s.e("load").disabled, true); assert.equal(s.e("google-retry").hidden, false);
  }
  completed.push("invalid mode, client ID, nonce, and login-CSRF fail before GIS loading");

  for (const failed of [reply({error: "FICTIONAL_ID_TOKEN_CANARY"}, 403), {ok: true, status: 200, json: async () => { throw new Error("FICTIONAL_ID_TOKEN_CANARY"); }}, reply({authenticated: true, csrf: "short"})]) {
    s = setup(fixture); await flush();
    s.setHandler(async (url, request, fallback) => url === "/review/auth/google" ? failed : fallback(url, request));
    await s.signIn(); await s.callback()({credential});
    assert.equal(loginPosts(s).length, 1); assert.equal(s.e("load").disabled, true); assert.equal(s.e("google-retry").hidden, false);
    assert.ok(!s.e("google-state").textContent.includes("CANARY")); assert.ok(!s.e("status").textContent.includes("CANARY"));
  }
  completed.push("refused, malformed, and invalid-CSRF login results remain locked and do not reflect or resend secrets");

  s = setup(fixture); await flush();
  s.setHandler(async (url, request, fallback) => {
    if (url === "/review/auth/google") { s.setSession(true); throw new TypeError("unconfirmed login"); }
    return fallback(url, request);
  });
  await s.signIn(); assert.equal(s.e("load").disabled, true);
  await s.e("google-retry").emit("click"); assert.equal(s.e("load").disabled, false);
  assert.equal(loginPosts(s).length, 1); completed.push("uncertain login reconciled through session GET without resubmitting ID token");

  s = setup(fixture, {restored: true}); await flush();
  s.setSession(false); s.setHandler(async (url, request, fallback) => url.includes("/review/api/") ? reply({error: credential}, 401) : fallback(url, request));
  await s.load(); assert.equal(s.e("load").disabled, true); assert.equal(s.e("google-retry").hidden, false);
  assert.ok(!s.e("status").textContent.includes(credential));
  await s.e("google-retry").emit("click"); await s.signIn();
  assert.equal(s.e("load").disabled, false); completed.push("expired API session locks review and supports explicit fresh sign-in");

  s = setup(fixture, {restored: true}); await flush(); await s.load(); await s.confirm();
  await s.emitGlobal("pagehide"); assert.equal(s.e("preview").hidden, true); assert.equal(s.e("load").disabled, true);
  await s.emitGlobal("pageshow", {persisted: true});
  assert.equal(s.e("load").disabled, false); assert.equal(s.e("preview").hidden, true); assert.equal(s.e("confirm-preview").checked, false);
  assert.equal(s.e("proposal-id").value, ""); assert.equal(decisionPosts(s).length, 0);
  completed.push("Back/Forward revalidates server session without restoring preview or confirmation");

  s = setup(fixture, {restored: true}); await flush();
  s.setHandler(async (url, request, fallback) => { if (url === "/review/auth/logout") throw new TypeError("unconfirmed logout"); return fallback(url, request); });
  await s.e("logout").emit("click");
  assert.equal(s.e("load").disabled, true); assert.equal(s.e("google-button").hidden, true); assert.equal(s.e("google-retry").hidden, false);
  assert.equal(s.calls.filter((call) => call.url === "/review/auth/logout").length, 1);
  await s.e("google-retry").emit("click"); assert.equal(s.e("load").disabled, false);
  completed.push("unconfirmed logout locks the page until explicit server-session check");

  s = setup(fixture, {restored: true}); await flush();
  s.setHandler(async (url, request, fallback) => url === "/review/auth/logout" ? reply({error: "stale_csrf"}, 401) : fallback(url, request));
  await s.e("logout").emit("click");
  assert.equal(s.e("load").disabled, true); assert.equal(s.e("google-retry").hidden, false);
  assert.ok(s.e("google-state").textContent.includes("could not be confirmed"));
  assert.equal(s.scripts.length, 0, "refused logout was mistaken for successful session removal");
  completed.push("logout 401 with a still-active cookie is unconfirmed rather than signed out");

  const checking = deferred(); let firstSession = true;
  s = setup(fixture, {restored: true, handler: async (url, request, fallback) => {
    if (url === "/review/auth/session" && firstSession) { firstSession = false; return checking.promise; }
    return fallback(url, request);
  }});
  await s.e("cancel").emit("click");
  checking.resolve(reply({authenticated: true, csrf})); await flush();
  assert.equal(s.e("load").disabled, true); assert.equal(s.e("google-button").hidden, false);
  assert.equal(s.calls.filter((call) => call.url === "/review/auth/logout").length, 1);
  completed.push("Cancel during initial session checking clears existing server authority and ignores its late response");

  s = setup(fixture, {restored: true}); await flush(); await s.load(); await s.confirm();
  const historyDecision = deferred();
  s.setHandler(async (url, request, fallback) => url.endsWith("/decision") ? historyDecision.promise : fallback(url, request));
  const navigatingDecision = s.e("approve").emit("click"); await flush();
  await s.emitGlobal("pagehide"); await s.emitGlobal("pageshow", {persisted: true});
  assert.equal(s.e("google-retry").hidden, false); assert.equal(s.e("google-retry").disabled, true);
  historyDecision.resolve(reply({...fixture.proposal, state: "approved"})); await navigatingDecision;
  assert.equal(s.e("google-retry").disabled, false);
  await s.e("google-retry").emit("click"); await s.load(); await s.confirm();
  assert.equal(s.e("approve").disabled, true); assert.equal(decisionPosts(s).length, 1);
  completed.push("history return during a decision waits for settlement and preserves the no-retry fence");

  s = setup(fixture); await flush(); const historyLogin = deferred();
  s.setHandler(async (url, request, fallback) => url === "/review/auth/google" ? historyLogin.promise : fallback(url, request));
  const navigatingLogin = s.callback()({credential}); await flush();
  await s.emitGlobal("pagehide"); await s.emitGlobal("pageshow", {persisted: true});
  assert.equal(s.e("load").disabled, true); assert.equal(s.initializations.length, 1);
  s.setSession(true); historyLogin.resolve(reply({authenticated: true, csrf})); await navigatingLogin; await flush();
  assert.equal(s.e("load").disabled, false); assert.equal(loginPosts(s).length, 1);
  completed.push("history return during login waits before reconciling the late cookie through session GET");

  s = setup(fixture, {restored: true, handler: async (url, request, fallback) => url === "/review/auth/session" ? reply({authenticated: true, csrf: "short"}) : fallback(url, request)}); await flush();
  assert.equal(s.e("load").disabled, true); assert.equal(s.scripts.length, 0);
  completed.push("malformed restored session cannot authorize review");

  s = setup(fixture, {autoScript: false}); await flush(); s.scripts[0].onerror(); await flush();
  assert.equal(s.e("google-retry").hidden, false); assert.equal(s.e("load").disabled, true); assert.equal(s.scripts[0].removed, true);
  completed.push("GIS script load failure remains locked without external fallback");

  s = setup(fixture, {mode: "bearer"}); await flush();
  assert.equal(s.calls.length, 0); assert.equal(s.scripts.length, 0); assert.equal(s.e("google-session").hidden, true);
  s.e("reviewer-token").value = "FICTIONAL_BEARER"; await s.e("session-form").emit("submit"); await s.load();
  assert.equal(s.calls[0].options.credentials, "omit"); assert.equal(s.calls[0].options.headers.Authorization, "Bearer FICTIONAL_BEARER");
  completed.push("bearer mode never loads Google and retains omit/header authentication");

  s = setup(fixture, {origin: "https://evil.example.invalid"}); await flush();
  await s.e("google-retry").emit("click"); await s.e("logout").emit("click"); await s.e("cancel").emit("click"); await s.load();
  assert.equal(s.calls.length, 0); assert.equal(s.scripts.length, 0); assert.equal(s.e("load").disabled, true);
  completed.push("foreign-origin session, retry, logout, Cancel, and preview remain disabled");

  for (const restored of [false, true]) {
    s = setup(fixture, {workflowState: "pending", restored}); await flush();
    if (!restored) { s.callback()({credential}); await flush(); }
    assert.match(s.e("google-state").textContent, /Reviewer signed in/);
    assert.equal(s.e("load").disabled, true);
    await s.load(); await s.confirm(); await s.e("approve").emit("click"); await s.e("deny").emit("click");
    assert.equal(s.e("approve").disabled, true); assert.equal(s.e("deny").disabled, true);
    assert.equal(s.calls.filter((call) => call.url.startsWith("/review/api/")).length, 0);
    assert.match(s.e("status").textContent, /Proposed actions are not available yet/);
    completed.push(restored ? "pending workflow restored login keeps all action controls disabled" : "pending workflow fresh login keeps all action controls disabled");
  }

  process.stdout.write(JSON.stringify({result: "google socket-free state checks passed", scenarios: completed}, null, 2) + "\n");
}
