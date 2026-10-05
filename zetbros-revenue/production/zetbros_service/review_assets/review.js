"use strict";
(() => {
  const byId = (id) => document.getElementById(id);
  const tokenInput = byId("reviewer-token");
  const idInput = byId("proposal-id");
  const confirmation = byId("confirm-preview");
  const status = byId("status");
  const panel = byId("preview");
  const pin = document.documentElement.dataset.reviewOrigin;
  const googleMode = document.documentElement.dataset.authMode === "google_oidc";
  const workflowPending = document.documentElement.dataset.workflowState === "pending";
  const sessionForm = byId("session-form");
  const googleSession = byId("google-session");
  const googleButton = byId("google-button");
  const googleRetry = byId("google-retry");
  const googleState = byId("google-state");
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const hash = /^[0-9a-f]{64}$/;
  let bearer = "";
  let reviewCsrf = "";
  let googleAuthenticated = false;
  let authGeneration = 0;
  let authController = null;
  let authBusy = false;
  let logoutBusy = false;
  let googleReady = false;
  let loginInFlight = null;
  let googleLibraryPromise = null;
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
  function authenticated() { return googleMode ? googleAuthenticated && !!reviewCsrf : !!bearer; }
  function safeOpaque(value) { return typeof value === "string" && /^[A-Za-z0-9_-]{16,256}$/.test(value); }
  function googleMessage(text) { if (googleState) googleState.textContent = text; }
  function expiredGoogleSession(response, version) {
    if (!googleMode || response.status !== 401 || version !== generation) return;
    googleAuthenticated = false;
    reviewCsrf = "";
    googleReady = false;
    if (googleRetry) googleRetry.hidden = false;
    googleMessage("Reviewer session expired. Check sign-in to continue.");
  }
  function reviewOptions(post = false) {
    const headers = {Accept: "application/json"};
    if (post) headers["Content-Type"] = "application/json";
    if (googleMode) {
      if (post) headers["X-Review-CSRF"] = reviewCsrf;
      return {headers, credentials: "same-origin"};
    }
    headers.Authorization = "Bearer " + bearer;
    return {headers, credentials: "omit"};
  }
  function pending() {
    return preview && preview.review_contract === "action_and_exact_wire_v1" && preview.wire_preview && preview.state === "pending" && !preview.execution && preview.consumed_at === null
      && Number.isSafeInteger(preview.expires_at) && Date.now() < preview.expires_at * 1000
      && !attempted.has(preview.id);
  }
  function controls() {
    byId("load").disabled = workflowPending || !authenticated() || deciding || logoutBusy || location.origin !== pin;
    if (byId("unlock")) byId("unlock").disabled = googleMode || deciding || location.origin !== pin;
    if (googleButton) googleButton.hidden = !googleMode || !googleReady || authBusy || logoutBusy || deciding || googleAuthenticated || location.origin !== pin;
    if (googleRetry) googleRetry.disabled = authBusy || logoutBusy || deciding || location.origin !== pin;
    byId("logout").disabled = googleMode && logoutBusy;
    const permitted = !workflowPending && authenticated() && !!pending() && confirmation.checked && !deciding && !loadController && !logoutBusy;
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
  function clearLocalSession() {
    bearer = "";
    reviewCsrf = "";
    googleAuthenticated = false;
    if (tokenInput) tokenInput.value = "";
    idInput.value = "";
    invalidate();
  }
  function stopGoogleAuth() {
    authGeneration += 1;
    if (authController) authController.abort();
    authController = null;
    authBusy = false;
    googleReady = false;
    if (googleRetry) googleRetry.hidden = true;
    // This only stops automatic browser sign-in, never revokes Google access.
    if (typeof google !== "undefined" && google.accounts && google.accounts.id) {
      try {
        google.accounts.id.cancel();
        google.accounts.id.disableAutoSelect();
      } catch (error) { /* Local session clearing must not depend on GIS. */ }
    }
  }
  function logout() {
    if (googleMode) return logoutGoogle();
    clearLocalSession();
    message(deciding ? "Session cleared. A decision request is still in flight and may have reached the ledger. Do not repeat it; load its status after it settles." : "Session and preview cleared. Enter a reviewer token to continue.");
  }

  function loadGoogleLibrary() {
    if (googleLibraryPromise) return googleLibraryPromise;
    googleLibraryPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://accounts.google.com/gsi/client";
      script.async = true;
      script.referrerPolicy = "no-referrer";
      script.onload = () => {
        if (typeof google !== "undefined" && google.accounts && google.accounts.id) resolve();
        else reject(new Error("google_library_unavailable"));
      };
      script.onerror = () => { script.remove(); reject(new Error("google_library_unavailable")); };
      document.head.appendChild(script);
    }).catch((error) => { googleLibraryPromise = null; throw error; });
    return googleLibraryPromise;
  }
  function authFailure(version, text) {
    if (version !== authGeneration) return;
    googleAuthenticated = false;
    reviewCsrf = "";
    authBusy = false;
    googleReady = false;
    googleMessage(text);
    if (googleRetry) googleRetry.hidden = false;
    controls();
  }
  function acceptGoogleSession(value) {
    if (!value || value.authenticated !== true || !safeOpaque(value.csrf)) throw new Error("invalid_session");
    googleAuthenticated = true;
    reviewCsrf = value.csrf;
    authBusy = false;
    googleReady = false;
    if (googleRetry) googleRetry.hidden = true;
    googleMessage(workflowPending ? "Reviewer signed in. Proposed actions are not available yet. Log out to clear this session." : "Reviewer signed in. Log out or Cancel to clear this session.");
    if (!deciding) message(workflowPending ? "Reviewer access verified. Proposed actions are not available yet. Email sending is disabled." : "Reviewer session verified. Load a proposal to review its exact preview.");
    controls();
  }
  async function prepareGoogleSignIn(version) {
    const controller = new AbortController();
    authController = controller;
    authBusy = true;
    googleReady = false;
    googleMessage("Preparing Google sign-in…");
    controls();
    const response = await fetch("/review/auth/bootstrap", {method: "GET", headers: {Accept: "application/json"}, credentials: "same-origin", cache: "no-store", redirect: "error", signal: controller.signal});
    const value = await responseValue(response);
    if (version !== authGeneration) return;
    if (!value || value.mode !== "google_oidc" || typeof value.client_id !== "string" || !/^[A-Za-z0-9._-]{1,200}\.apps\.googleusercontent\.com$/.test(value.client_id) || !safeOpaque(value.nonce) || !safeOpaque(value.csrf)) throw new Error("invalid_bootstrap");
    await loadGoogleLibrary();
    if (version !== authGeneration) return;
    authController = null;
    googleButton.textContent = "";
    google.accounts.id.initialize({client_id: value.client_id, nonce: value.nonce, auto_select: false,
      callback: (result) => completeGoogleLogin(result, version, value.csrf)});
    google.accounts.id.renderButton(googleButton, {type: "standard", theme: "outline", size: "large", text: "signin_with"});
    authBusy = false;
    googleReady = true;
    googleMessage("Sign in with an approved reviewer account.");
    if (!deciding) message(workflowPending ? "Sign in with your approved Google account. Proposed actions are not available yet." : "Sign in with Google, then load a proposal.");
    controls();
  }
  async function checkGoogleSession() {
    if (!googleMode || location.origin !== pin) return;
    if (authBusy) return;
    if (logoutBusy || deciding) {
      if (googleRetry) googleRetry.hidden = false;
      googleMessage("Wait for the in-flight request to settle, then check sign-in.");
      controls();
      return;
    }
    const activeLogin = loginInFlight;
    stopGoogleAuth();
    clearLocalSession();
    const version = authGeneration;
    const controller = new AbortController();
    authController = controller;
    authBusy = true;
    googleMessage("Checking reviewer session…");
    controls();
    try {
      if (activeLogin) await activeLogin;
      if (version !== authGeneration) return;
      const response = await fetch("/review/auth/session", {method: "GET", headers: {Accept: "application/json"}, credentials: "same-origin", cache: "no-store", redirect: "error", signal: controller.signal});
      if (version !== authGeneration) return;
      if (response.status === 401) { await prepareGoogleSignIn(version); return; }
      const value = await responseValue(response);
      if (version !== authGeneration) return;
      if (value && value.authenticated === false) { await prepareGoogleSignIn(version); return; }
      acceptGoogleSession(value);
    } catch (error) {
      authFailure(version, "Reviewer session could not be verified. Check sign-in to try again.");
    } finally {
      if (version === authGeneration) authController = null;
    }
  }
  function completeGoogleLogin(result, version, loginCsrf) {
    if (version !== authGeneration || authBusy || logoutBusy || deciding || googleAuthenticated || !googleReady || location.origin !== pin) return;
    if (!result || typeof result.credential !== "string" || !result.credential || result.credential.length > 8192 || /\s/.test(result.credential)) {
      authFailure(version, "Google sign-in did not provide a valid response. Check sign-in to start again.");
      return;
    }
    authBusy = true;
    googleReady = false;
    googleMessage("Verifying reviewer access…");
    controls();
    // The GIS ID token is submitted once to this origin. It is never used as an
    // API bearer, saved, decoded for authority, or sent to any other endpoint.
    const operation = (async () => {
      try {
        const response = await fetch("/review/auth/google", {method: "POST", headers: {Accept: "application/json", "Content-Type": "application/json", "X-Login-CSRF": loginCsrf}, body: JSON.stringify({credential: result.credential}), credentials: "same-origin", cache: "no-store", redirect: "error"});
        const value = await responseValue(response);
        if (version !== authGeneration) return;
        acceptGoogleSession(value);
      } catch (error) {
        authFailure(version, "Google sign-in was refused or could not be confirmed. Check sign-in before continuing; the sign-in response will not be resubmitted.");
      }
    })();
    loginInFlight = operation;
    operation.finally(() => { if (loginInFlight === operation) loginInFlight = null; });
    return operation;
  }
  async function logoutGoogle() {
    if (logoutBusy) return;
    if (location.origin !== pin) { stopGoogleAuth(); clearLocalSession(); return; }
    const csrf = reviewCsrf;
    const activeLogin = loginInFlight;
    stopGoogleAuth();
    clearLocalSession();
    const version = authGeneration;
    logoutBusy = true;
    controls();
    googleMessage(activeLogin ? "Waiting for the sign-in request to settle before clearing the server session…" : "Clearing reviewer session…");
    message(deciding ? "Session cleared. A decision request is still in flight and may have reached the ledger. Do not repeat it; load its status after it settles." : "Preview cleared. Clearing the server reviewer session…");
    try {
      let logoutCsrf = csrf;
      // A login response can set a cookie after Cancel. Wait without aborting
      // its POST, then resolve and clear that session before permitting login.
      if (activeLogin || !logoutCsrf) {
        if (activeLogin) await activeLogin;
        const response = await fetch("/review/auth/session", {method: "GET", headers: {Accept: "application/json"}, credentials: "same-origin", cache: "no-store", redirect: "error"});
        if (response.status !== 401) {
          const value = await responseValue(response);
          if (value && value.authenticated === true && safeOpaque(value.csrf)) logoutCsrf = value.csrf;
          else if (!value || value.authenticated !== false) throw new Error("invalid_session");
        }
      }
      if (logoutCsrf) {
        const response = await fetch("/review/auth/logout", {method: "POST", headers: {Accept: "application/json", "X-Review-CSRF": logoutCsrf}, credentials: "same-origin", cache: "no-store", redirect: "error"});
        if (!response.ok) {
          if (response.status !== 401) throw new Error("logout_unconfirmed");
          // A refused logout can also mean a stale CSRF value. Only a fresh
          // unauthenticated session check establishes that authority is gone.
          const current = await fetch("/review/auth/session", {method: "GET", headers: {Accept: "application/json"}, credentials: "same-origin", cache: "no-store", redirect: "error"});
          if (current.status !== 401) throw new Error("logout_unconfirmed");
        }
      }
      if (version !== authGeneration) return;
      logoutBusy = false;
      if (!deciding) message("Reviewer session and preview cleared.");
      await prepareGoogleSignIn(version);
    } catch (error) {
      if (version !== authGeneration) return;
      logoutBusy = false;
      authFailure(version, "Log out could not be confirmed. This page is locked. Check sign-in to verify the server session before continuing.");
      if (!deciding) message("Preview cleared. The server session's log out outcome is unconfirmed.");
    } finally {
      logoutBusy = false;
      controls();
    }
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
  if (sessionForm && tokenInput) sessionForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (googleMode || deciding || location.origin !== pin) return;
    invalidate();
    bearer = tokenInput.value;
    tokenInput.value = "";
    if (!bearer || bearer.length > 16377 || /\s/.test(bearer)) { bearer = ""; message("Enter one existing access token without the Bearer prefix or whitespace."); }
    else message("Token held in page memory. Load a proposal to verify reviewer access.");
    controls();
  });
  if (tokenInput) tokenInput.addEventListener("input", () => { if (bearer) { bearer = ""; invalidate(); message("Session cleared while replacing the token."); } });
  byId("logout").addEventListener("click", logout);
  byId("cancel").addEventListener("click", logout);
  if (googleRetry) googleRetry.addEventListener("click", checkGoogleSession);
  idInput.addEventListener("input", () => { if (workflowPending) return; invalidate(); message(deciding ? "A decision is in flight. Changing this field does not cancel it." : "Proposal changed. Load a fresh preview before deciding."); });
  confirmation.addEventListener("change", controls);
  byId("proposal-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (workflowPending || !authenticated() || deciding || logoutBusy || location.origin !== pin) return;
    const id = idInput.value;
    invalidate();
    if (!uuid.test(id)) { message("Enter a complete proposal UUID."); return; }
    const version = generation;
    const controller = new AbortController();
    loadController = controller;
    controls();
    message("Loading the stored exact preview…");
    try {
      const response = await fetch("/review/api/proposals/" + encodeURIComponent(id), {method: "GET", ...reviewOptions(), cache: "no-store", redirect: "error", signal: controller.signal});
      expiredGoogleSession(response, version);
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
    if (workflowPending || !authenticated() || !pending() || !confirmation.checked || deciding || loadController || logoutBusy || location.origin !== pin) return;
    const shown = preview;
    const version = generation;
    attempted.add(shown.id);
    deciding = true;
    confirmation.checked = false;
    controls();
    message("Recording the " + decision + " decision. Do not repeat this request…");
    try {
      const response = await fetch("/review/api/proposals/" + encodeURIComponent(shown.id) + "/decision", {method: "POST", ...reviewOptions(true), body: JSON.stringify({digest: shown.digest, decision}), cache: "no-store", redirect: "error"});
      expiredGoogleSession(response, version);
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
  addEventListener("pagehide", () => {
    if (googleMode) { stopGoogleAuth(); clearLocalSession(); }
    else logout();
  });
  addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    if (googleMode) { stopGoogleAuth(); clearLocalSession(); checkGoogleSession(); }
    else logout();
  });
  if (sessionForm) sessionForm.hidden = googleMode;
  if (tokenInput && googleMode) { tokenInput.value = ""; tokenInput.disabled = true; }
  if (googleSession) googleSession.hidden = !googleMode;
  if (location.origin !== pin) {
    if (googleMode) stopGoogleAuth();
    clearLocalSession();
    message("This page is outside the configured trusted review origin. Reviewer access is disabled.");
  } else if (googleMode) checkGoogleSession();
  controls();
})();
