"use strict";
(() => {
  const byId = (id) => document.getElementById(id);
  const tokenInput = byId("reviewer-token");
  const idInput = byId("proposal-id");
  const confirmation = byId("confirm-preview");
  const status = byId("status");
  const panel = byId("preview");
  const pin = document.documentElement.dataset.reviewOrigin;
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const hash = /^[0-9a-f]{64}$/;
  let bearer = "";
  let generation = 0;
  let preview = null;
  let loadController = null;
  let deciding = false;
  let expiredTimer = null;
  // A submitted/possibly submitted decision is never retried in this page.
  // This survives Cancel/Log out in memory and is never persisted or transmitted.
  const attempted = new Set();

  class SafeResponseError extends Error {}
  function message(text) { status.textContent = text; }
  function pending() {
    return preview && preview.review_contract === "action_and_exact_wire_v1" && preview.wire_preview && preview.state === "pending" && !preview.execution && preview.consumed_at === null
      && Number.isSafeInteger(preview.expires_at) && Date.now() < preview.expires_at * 1000
      && !attempted.has(preview.id);
  }
  function controls() {
    byId("load").disabled = !bearer || deciding || location.origin !== pin;
    byId("unlock").disabled = deciding || location.origin !== pin;
    const permitted = !!bearer && !!pending() && confirmation.checked && !deciding && !loadController;
    byId("approve").disabled = !permitted;
    byId("deny").disabled = !permitted;
  }
  function clearPreview() {
    preview = null;
    confirmation.checked = false;
    panel.hidden = true;
    for (const id of ["identity", "reply-body", "action", "wire", "wire-body", "ledger"]) byId(id).textContent = "";
    byId("wire-section").hidden = true;
    if (expiredTimer !== null) clearTimeout(expiredTimer);
    expiredTimer = null;
    controls();
  }
  function invalidate() {
    generation += 1;
    if (loadController) loadController.abort();
    loadController = null;
    clearPreview();
  }
  function logout() {
    bearer = "";
    tokenInput.value = "";
    idInput.value = "";
    invalidate();
    message(deciding ? "Session cleared. A decision request is still in flight and may have reached the ledger. Do not repeat it; load its status after it settles." : "Session and preview cleared. Enter a reviewer token to continue.");
  }
  function showIdentity(label, value) {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = typeof value === "string" ? value : JSON.stringify(value);
    byId("identity").append(term, detail);
  }
  function display(value) {
    if (!value || !uuid.test(value.id) || !hash.test(value.digest) || !value.payload || typeof value.payload.body !== "string") throw new Error("invalid_preview");
    if (value.review_contract !== "action_and_exact_wire_v1" || !value.wire_preview || !hash.test(value.action_digest) || !hash.test(value.wire_preview_digest) || typeof value.wire_preview.body !== "string") throw new Error("invalid_preview");
    preview = value;
    const action = value.payload;
    for (const [label, field] of [["Tenant", "tenant_id"], ["Connector", "connector_id"], ["Account", "account_id"], ["Source ID", "source_id"], ["Source version", "source_version"], ["Source fingerprint", "source_fingerprint"], ["Sender", "sender"], ["To", "to"], ["Subject", "subject"], ["Operation", "operation"], ["Operation key", "operation_key"], ["Policy", "policy_version"]]) showIdentity(label, action[field]);
    showIdentity("Proposal ID", value.id);
    showIdentity("Complete review digest", value.digest);
    showIdentity("Review contract", value.review_contract);
    showIdentity("Ledger state", value.state);
    showIdentity("Expires (UTC)", new Date(value.expires_at * 1000).toISOString());
    byId("reply-body").textContent = action.body;
    byId("action").textContent = JSON.stringify(action, null, 2);
    if (value.wire_preview) {
      showIdentity("Action digest", value.action_digest);
      showIdentity("Wire-preview digest", value.wire_preview_digest);
      byId("wire").textContent = JSON.stringify(value.wire_preview, null, 2);
      byId("wire-body").textContent = value.wire_preview.body;
      byId("wire-section").hidden = false;
    }
    const ledger = Object.fromEntries(Object.entries(value).filter(([key]) => !["payload", "wire_preview"].includes(key)));
    byId("ledger").textContent = JSON.stringify(ledger, null, 2);
    panel.hidden = false;
    confirmation.checked = false;
    const delay = Math.max(0, value.expires_at * 1000 - Date.now());
    expiredTimer = setTimeout(() => { controls(); if (preview === value) message("This preview has expired. No decision can be made from it."); }, Math.min(delay, 2147483647));
    controls();
  }
  async function responseValue(response) {
    // Never display raw error bodies: a proxy/server failure could reflect secrets.
    if (!response.ok) {
      const labels = {401: "Reviewer authentication failed or expired.", 403: "Reviewer access or trusted origin was refused.", 404: "Proposal was not found.", 409: "Exact wire preview is unavailable, or this proposal changed, expired, or already has a decision. Load its current state.", 429: "Rate limit reached. Wait before loading again."};
      throw new SafeResponseError(labels[response.status] || "The request was refused. Load the current state before taking another action.");
    }
    return response.json();
  }
  byId("session-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (deciding || location.origin !== pin) return;
    invalidate();
    bearer = tokenInput.value;
    tokenInput.value = "";
    if (!bearer || bearer.length > 16377 || /\s/.test(bearer)) { bearer = ""; message("Enter one existing access token without the Bearer prefix or whitespace."); }
    else message("Token held in page memory. Load a proposal to verify reviewer access.");
    controls();
  });
  tokenInput.addEventListener("input", () => { if (bearer) { bearer = ""; invalidate(); message("Session cleared while replacing the token."); } });
  byId("logout").addEventListener("click", logout);
  byId("cancel").addEventListener("click", logout);
  idInput.addEventListener("input", () => { invalidate(); message(deciding ? "A decision is in flight. Changing this field does not cancel it." : "Proposal changed. Load a fresh preview before deciding."); });
  confirmation.addEventListener("change", controls);
  byId("proposal-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!bearer || deciding || location.origin !== pin) return;
    const id = idInput.value;
    invalidate();
    if (!uuid.test(id)) { message("Enter a complete proposal UUID."); return; }
    const version = generation;
    const controller = new AbortController();
    loadController = controller;
    controls();
    message("Loading the stored exact preview…");
    try {
      const response = await fetch("/review/api/proposals/" + encodeURIComponent(id), {method: "GET", headers: {Authorization: "Bearer " + bearer, Accept: "application/json"}, credentials: "omit", cache: "no-store", redirect: "error", signal: controller.signal});
      const value = await responseValue(response);
      if (version !== generation) return;
      if (value.id.toLowerCase() !== id.toLowerCase()) throw new Error("invalid_preview");
      loadController = null;
      display(value);
      message(attempted.has(value.id) ? "A decision was already attempted for this proposal in this page. Its current ledger state is shown. Do not repeat an uncertain request." : pending() ? "Check the full preview, then confirm before deciding." : "Current state shown. This proposal is no longer eligible for a decision.");
    } catch (error) {
      if (version !== generation || (error && error.name === "AbortError")) return;
      clearPreview();
      message(error instanceof SafeResponseError ? error.message : "Exact wire preview could not be verified or loaded. No decision was sent.");
    } finally {
      if (version === generation) { loadController = null; controls(); }
    }
  });
  async function decide(decision) {
    if (!bearer || !pending() || !confirmation.checked || deciding || loadController || location.origin !== pin) return;
    const shown = preview;
    const version = generation;
    attempted.add(shown.id);
    deciding = true;
    confirmation.checked = false;
    controls();
    message("Recording the " + decision + " decision. Do not repeat this request…");
    try {
      const response = await fetch("/review/api/proposals/" + encodeURIComponent(shown.id) + "/decision", {method: "POST", headers: {Authorization: "Bearer " + bearer, Accept: "application/json", "Content-Type": "application/json"}, body: JSON.stringify({digest: shown.digest, decision}), credentials: "omit", cache: "no-store", redirect: "error"});
      const value = await responseValue(response);
      if (version !== generation) return;
      if (!value || value.id !== shown.id || value.digest !== shown.digest || value.state !== (decision === "approve" ? "approved" : "denied")) throw new Error("invalid_decision_response");
      clearPreview();
      display(value);
      message("Decision recorded. Current ledger state: " + value.state + ". Approval is not evidence of sending or delivery.");
    } catch (error) {
      if (version !== generation) return;
      clearPreview();
      message("The decision outcome is unconfirmed. It may have reached the ledger. Do not repeat it. Load current status; this page will not submit another decision for this proposal.");
    } finally {
      deciding = false;
      controls();
      if (version !== generation) message("The in-flight decision has settled, but its outcome was not retained after the session changed. Load authoritative status; do not repeat an uncertain request.");
    }
  }
  byId("approve").addEventListener("click", () => decide("approve"));
  byId("deny").addEventListener("click", () => decide("deny"));
  addEventListener("pagehide", logout);
  addEventListener("pageshow", (event) => { if (event.persisted) logout(); });
  if (location.origin !== pin) { logout(); message("This page is outside the configured trusted review origin. Reviewer access is disabled."); }
  controls();
})();
