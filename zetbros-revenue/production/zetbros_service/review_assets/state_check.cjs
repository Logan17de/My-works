/* Socket-free behavior harness. Runs the shipped script in a minimal DOM model.
   This is a state-machine check, not a claim of rendered-browser verification. */
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
  async emit(name, event = {}) {
    event.preventDefault = () => {};
    const promises = (this.listeners.get(name) || []).map((fn) => fn(event));
    await Promise.all(promises);
  }
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
const clone = (value) => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; };

function setup(fixture, origin = fixture.origin) {
  const elements = new Map();
  const html = fs.readFileSync(fixture.assets + "/review.html", "utf8");
  for (const match of html.matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], new Element(match[1]));
  elements.get("preview").hidden = true; elements.get("wire-section").hidden = true;
  const document = {documentElement: {dataset: {reviewOrigin: fixture.origin}}, getElementById: (id) => elements.get(id), createElement: (tag) => new Element(tag)};
  const globals = new Map(); const calls = []; const timers = new Map(); let timerNumber = 0;
  let handler = async () => ({ok: true, status: 200, json: async () => clone(fixture.proposal)});
  const context = vm.createContext({document, location: {origin}, AbortController, Set, Date, Error, TypeError, Object, Number, JSON,
    setTimeout: (fn) => { const id = ++timerNumber; timers.set(id, fn); return id; }, clearTimeout: (id) => timers.delete(id),
    addEventListener: (event, fn) => { if (!globals.has(event)) globals.set(event, []); globals.get(event).push(fn); },
    fetch: (url, options) => { calls.push({url, options: {...options, headers: {...options.headers}}}); return handler(url, options); }});
  vm.runInContext(fs.readFileSync(fixture.assets + "/review.js", "utf8"), context, {filename: "review.js"});
  const e = (id) => elements.get(id);
  const env = {e, calls, timers, context, setHandler: (fn) => { handler = fn; },
    emitGlobal: async (event, data = {}) => { for (const fn of globals.get(event) || []) await fn(data); },
    unlock: async (token = "fictional_memory_only_token") => { e("reviewer-token").value = token; await e("session-form").emit("submit"); },
    input: async (value) => { e("proposal-id").value = value; await e("proposal-id").emit("input"); },
    load: async (id = fixture.proposal.id) => { await env.input(id); await e("proposal-form").emit("submit"); },
    confirm: async () => { e("confirm-preview").checked = true; await e("confirm-preview").emit("change"); }};
  return env;
}

async function main(fixture) {
  let s = setup(fixture);
  assert.equal(s.e("load").disabled, true);
  await s.unlock();
  assert.equal(s.e("reviewer-token").value, "");
  await s.load();
  assert.equal(s.e("preview").hidden, false);
  assert.equal(s.e("reply-body").textContent, fixture.proposal.payload.body);
  assert.equal(s.e("wire-body").textContent, fixture.proposal.wire_preview.body);
  assert.deepEqual(JSON.parse(s.e("action").textContent), fixture.proposal.payload);
  assert.deepEqual(JSON.parse(s.e("wire").textContent), fixture.proposal.wire_preview);
  assert.equal(s.e("reply-body").children.length, 0, "mail became DOM elements");
  assert.equal(s.e("approve").disabled, true);
  assert.ok(s.e("identity").textContent.includes(fixture.proposal.digest));
  await s.confirm(); assert.equal(s.e("approve").disabled, false);
  const response = deferred();
  s.setHandler(async (url) => {
    assert.ok(url.endsWith("/decision"));
    return response.promise;
  });
  const decision = s.e("approve").emit("click"); await tick();
  assert.equal(s.e("approve").disabled, true);
  await s.e("approve").emit("click");
  assert.equal(s.calls.filter((call) => call.options.method === "POST").length, 1);
  assert.deepEqual(JSON.parse(s.calls.at(-1).options.body), {digest: fixture.proposal.digest, decision: "approve"});
  response.resolve({ok: true, status: 200, json: async () => ({...clone(fixture.proposal), state: "approved"})});
  await decision;
  assert.equal(s.e("approve").disabled, true);
  assert.ok(s.e("status").textContent.includes("Decision recorded"));
  await s.e("logout").emit("click");
  for (const id of ["identity", "reply-body", "action", "wire", "wire-body", "ledger"]) assert.equal(s.e(id).textContent, "");
  assert.equal(s.e("reviewer-token").value, ""); assert.equal(s.e("proposal-id").value, "");
  assert.equal(s.e("preview").hidden, true); assert.equal(s.e("load").disabled, true);
  // Editing a token or source clears old authority/preview immediately.
  s = setup(fixture); await s.unlock(); await s.load(); await s.confirm();
  s.e("reviewer-token").value = "replacement"; await s.e("reviewer-token").emit("input");
  assert.equal(s.e("load").disabled, true); assert.equal(s.e("preview").hidden, true);
  await s.unlock(); await s.load(); await s.confirm(); await s.input("changed-id");
  assert.equal(s.e("preview").hidden, true); assert.equal(s.e("approve").disabled, true);
  // Raced GETs: an older response ignored even if abort is ignored by transport.
  s = setup(fixture); await s.unlock();
  const old = deferred(); const second = clone(fixture.proposal);
  second.id = "00000000-0000-0000-0000-000000000002"; second.payload.body = "Newer exact preview";
  s.setHandler(async (url) => url.endsWith(second.id) ? {ok: true, status: 200, json: async () => second} : old.promise);
  await s.input(fixture.proposal.id); const firstLoad = s.e("proposal-form").emit("submit"); await tick();
  await s.load(second.id);
  old.resolve({ok: true, status: 200, json: async () => fixture.proposal}); await firstLoad;
  assert.equal(s.e("reply-body").textContent, "Newer exact preview");
  assert.ok(s.e("identity").textContent.includes(second.id));
  assert.equal(s.calls[0].options.signal.aborted, true);
  // Cancel a GET: a late result cannot repopulate cleared mail/token.
  s = setup(fixture); await s.unlock(); const delayed = deferred();
  s.setHandler(async () => delayed.promise); await s.input(fixture.proposal.id);
  const interrupted = s.e("proposal-form").emit("submit"); await tick();
  await s.e("cancel").emit("click");
  delayed.resolve({ok: true, status: 200, json: async () => fixture.proposal}); await interrupted;
  assert.equal(s.e("preview").hidden, true); assert.equal(s.e("load").disabled, true);
  // Cancel a POST: do not abort/claim cancellation; fence reissue after it settles.
  s = setup(fixture); await s.unlock(); await s.load(); await s.confirm();
  const uncertain = deferred(); s.setHandler(async () => uncertain.promise);
  const inFlight = s.e("approve").emit("click"); await tick();
  assert.equal(s.calls.at(-1).options.signal, undefined, "POST was aborted despite possible ledger commit");
  await s.e("cancel").emit("click");
  assert.ok(s.e("status").textContent.includes("still in flight"));
  assert.equal(s.e("unlock").disabled, true);
  uncertain.reject(new TypeError("unavailable")); await inFlight;
  assert.equal(s.e("preview").hidden, true);
  s.setHandler(async () => ({ok: true, status: 200, json: async () => fixture.proposal}));
  await s.unlock(); await s.load(); await s.confirm(); await s.e("approve").emit("click");
  assert.equal(s.e("approve").disabled, true);
  assert.equal(s.calls.filter((call) => call.options.method === "POST").length, 1);
  // An uncertain POST without cancellation is also never automatically retried.
  s = setup(fixture); await s.unlock(); await s.load(); await s.confirm();
  s.setHandler(async () => { throw new TypeError("connection unavailable"); });
  await s.e("deny").emit("click");
  assert.ok(s.e("status").textContent.includes("outcome is unconfirmed"));
  assert.equal(s.e("preview").hidden, true);
  s.setHandler(async () => ({ok: true, status: 200, json: async () => fixture.proposal}));
  await s.load(); await s.confirm(); await s.e("deny").emit("click");
  assert.equal(s.e("deny").disabled, true);
  assert.equal(s.calls.filter((call) => call.options.method === "POST").length, 1);
  // An unrelated successful POST response never confirms the shown decision.
  s = setup(fixture); await s.unlock(); await s.load(); await s.confirm();
  s.setHandler(async () => ({ok: true, status: 200, json: async () => ({...fixture.proposal, id: "00000000-0000-0000-0000-000000000099", state: "approved"})}));
  await s.e("approve").emit("click");
  assert.ok(s.e("status").textContent.includes("outcome is unconfirmed"));
  assert.equal(s.e("preview").hidden, true);
  // Response errors do not reflect any server-provided bearer/body material.
  s = setup(fixture); await s.unlock();
  s.setHandler(async () => ({ok: false, status: 401, json: async () => ({secret: "TOKEN_CANARY"})}));
  await s.load(); assert.equal(s.e("preview").hidden, true);
  assert.ok(!s.e("status").textContent.includes("CANARY"));
  // Malformed success JSON must never reflect raw body/parser snippets.
  s = setup(fixture); await s.unlock();
  s.setHandler(async () => ({ok: true, status: 200, json: async () => { throw new Error("Unexpected token BODY_CANARY TOKEN_CANARY"); }}));
  await s.load(); assert.equal(s.e("preview").hidden, true);
  assert.ok(!s.e("status").textContent.includes("CANARY"));
  assert.equal(s.e("approve").disabled, true);
  // Action-only ledgers cannot masquerade as an exact-wire preview.
  s = setup(fixture); await s.unlock(); const actionOnly = clone(fixture.proposal);
  delete actionOnly.wire_preview; delete actionOnly.review_contract;
  s.setHandler(async () => ({ok: true, status: 200, json: async () => actionOnly}));
  await s.load(); await s.confirm(); assert.equal(s.e("preview").hidden, true); assert.equal(s.e("approve").disabled, true);
  // Invalid/mismatched preview cannot leave half-populated stale approval.
  s = setup(fixture); await s.unlock();
  s.setHandler(async () => ({ok: true, status: 200, json: async () => ({...fixture.proposal, id: "wrong"})}));
  await s.load(); assert.equal(s.e("preview").hidden, true); assert.equal(s.e("approve").disabled, true);
  // Expired preview and restored history never expose active approval.
  s = setup({...fixture, proposal: {...fixture.proposal, expires_at: Math.floor(Date.now()/1000)-1}});
  await s.unlock(); await s.load(); await s.confirm(); assert.equal(s.e("approve").disabled, true);
  s = setup(fixture); await s.unlock(); await s.load(); await s.confirm();
  await s.emitGlobal("pagehide"); assert.equal(s.e("preview").hidden, true); assert.equal(s.e("load").disabled, true);
  await s.emitGlobal("pageshow", {persisted: true}); assert.equal(s.e("reviewer-token").value, "");
  // Foreign location never activates even with a token and direct form dispatch.
  s = setup(fixture, "https://evil.example.invalid"); await s.unlock(); await s.load();
  assert.equal(s.e("load").disabled, true); assert.equal(s.calls.length, 0);
  process.stdout.write("socket-free state checks passed: full text/digest, confirmation, repeat/stale/racing GETs, Cancel/Log out, uncertain POST, history, origin\n");
}
