"use strict";
(() => {
  const byId = (id) => document.getElementById(id);
  const pin = document.documentElement.dataset.origin;
  const start = byId("start-proof");
  const cancel = byId("cancel-proof");
  const button = byId("google-button");
  const status = byId("proof-status");
  const resultPanel = byId("proof-result");
  const subject = byId("proof-subject");
  let generation = 0;
  let phase = "idle";
  let challengeCsrf = "";
  let bootstrapInFlight = null;
  let proofInFlight = null;
  let cancelInFlight = null;
  let libraryPromise = null;

  function trusted() { return location.origin === pin; }
  function message(text) { status.textContent = text; }
  function opaque(value) { return typeof value === "string" && /^[A-Za-z0-9_-]{43}$/.test(value); }
  function controls() {
    const networkBusy = !!bootstrapInFlight || !!proofInFlight || !!cancelInFlight;
    start.disabled = !trusted() || networkBusy || ["bootstrap", "loading", "proving", "cancelling"].includes(phase);
    cancel.disabled = !trusted() || phase === "cancelling" || (phase === "idle" && !networkBusy && resultPanel.hidden && !challengeCsrf);
    button.hidden = !trusted() || phase !== "ready";
  }
  function clearResult() {
    subject.textContent = "";
    byId("proof-authority").textContent = "Authority: none";
    resultPanel.hidden = true;
  }
  function stopGoogle() {
    if (typeof google !== "undefined" && google.accounts && google.accounts.id) {
      try { google.accounts.id.cancel(); google.accounts.id.disableAutoSelect(); }
      catch (error) { /* Clearing this page does not depend on Google. */ }
    }
    button.textContent = "";
  }
  function clearPage() {
    generation += 1;
    challengeCsrf = "";
    phase = "idle";
    stopGoogle();
    clearResult();
    message("Proof cleared. Start a fresh proof when you're ready. No application authority was granted.");
    controls();
  }
  function loadGoogleLibrary() {
    if (libraryPromise) return libraryPromise;
    libraryPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      let settled = false;
      const timer = setTimeout(() => fail(), 15000);
      function fail() {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        script.remove();
        reject(new Error("identity_library_unavailable"));
      }
      script.src = "https://accounts.google.com/gsi/client";
      script.async = true;
      script.referrerPolicy = "no-referrer";
      script.onerror = fail;
      script.onload = () => {
        if (settled) return;
        if (typeof google === "undefined" || !google.accounts || !google.accounts.id) { fail(); return; }
        settled = true;
        clearTimeout(timer);
        resolve();
      };
      document.head.appendChild(script);
    }).catch((error) => { libraryPromise = null; throw error; });
    return libraryPromise;
  }
  async function responseValue(response) {
    // Error bodies and parser messages may reflect credentials. Never show them.
    if (!response.ok) throw new Error("identity_request_refused");
    return response.json();
  }
  function bootstrapRequest() {
    const operation = (async () => {
      const response = await fetch("/identity/bootstrap", {method: "GET", headers: {Accept: "application/json"}, credentials: "same-origin", cache: "no-store", redirect: "error"});
      const value = await responseValue(response);
      if (!value || typeof value.client_id !== "string" || !/^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$/.test(value.client_id) || value.client_id.length > 256 || !opaque(value.nonce) || !opaque(value.csrf)) throw new Error("invalid_identity_challenge");
      return value;
    })();
    bootstrapInFlight = operation;
    const settled = () => { if (bootstrapInFlight === operation) bootstrapInFlight = null; controls(); };
    operation.then(settled, settled);
    return operation;
  }
  async function startProof() {
    if (!trusted() || bootstrapInFlight || proofInFlight || cancelInFlight || ["bootstrap", "loading", "proving", "cancelling"].includes(phase)) return;
    generation += 1;
    const version = generation;
    phase = "bootstrap";
    challengeCsrf = "";
    clearResult();
    stopGoogle();
    controls();
    message("Preparing a one-time identity challenge…");
    try {
      const value = await bootstrapRequest();
      if (version !== generation) return;
      challengeCsrf = value.csrf;
      phase = "loading";
      controls();
      message("Loading Google sign-in for this identity proof…");
      await loadGoogleLibrary();
      if (version !== generation) return;
      google.accounts.id.initialize({client_id: value.client_id, nonce: value.nonce, auto_select: false,
        callback: (response) => submitProof(response, version, value.csrf)});
      google.accounts.id.renderButton(button, {type: "standard", theme: "outline", size: "large", text: "signin_with"});
      phase = "ready";
      message("Choose your Google account to prove its immutable subject. This grants no application authority.");
      controls();
    } catch (error) {
      if (version !== generation) return;
      phase = "idle";
      message("The identity challenge could not be prepared. No identity result is shown. Start a fresh proof to try again.");
      controls();
    }
  }
  function submitProof(response, version, csrf) {
    if (!trusted() || version !== generation || phase !== "ready" || proofInFlight || cancelInFlight || csrf !== challengeCsrf) return;
    if (!response || typeof response.credential !== "string" || !response.credential || response.credential.length > 8192 || /\s/.test(response.credential)) {
      phase = "idle";
      message("Google did not provide a valid proof response. Start a fresh proof; nothing was submitted.");
      controls();
      return;
    }
    phase = "proving";
    controls();
    message("Verifying the one-time identity proof. This response will be submitted once…");
    // The ID token goes only to this dedicated proof endpoint. It is never
    // decoded for local authority, rendered, persisted, or used as an API bearer.
    const operation = (async () => {
      try {
        const reply = await fetch("/identity/proof", {method: "POST", headers: {Accept: "application/json", "Content-Type": "application/json", "X-Proof-CSRF": csrf}, body: JSON.stringify({credential: response.credential}), credentials: "same-origin", cache: "no-store", redirect: "error"});
        const value = await responseValue(reply);
        if (version !== generation) return;
        if (!value || Object.keys(value).sort().join(",") !== "authority,subject,verified" || value.verified !== true || value.authority !== "none" || typeof value.subject !== "string" || !/^[\x21-\x7e]{1,255}$/.test(value.subject) || value.subject.includes("@") || value.subject.includes(response.credential)) throw new Error("invalid_identity_proof");
        subject.textContent = value.subject;
        byId("proof-authority").textContent = "Authority: none";
        resultPanel.hidden = false;
        challengeCsrf = "";
        phase = "idle";
        message("Identity proof verified. The subject below has no application authority.");
      } catch (error) {
        if (version !== generation) return;
        challengeCsrf = "";
        phase = "idle";
        clearResult();
        message("The identity proof was refused or its result could not be confirmed. No result is shown. Start a fresh proof; this response will not be resubmitted.");
      } finally { controls(); }
    })();
    proofInFlight = operation;
    operation.then(() => { if (proofInFlight === operation) proofInFlight = null; controls(); });
    return operation;
  }
  function cancelProof() {
    if (cancelInFlight) return cancelInFlight;
    if (!trusted()) { clearPage(); return; }
    const previousCsrf = challengeCsrf;
    const activeBootstrap = bootstrapInFlight;
    const activeProof = proofInFlight;
    generation += 1;
    const version = generation;
    phase = "cancelling";
    challengeCsrf = "";
    stopGoogle();
    clearResult();
    message(activeProof ? "Proof cleared. Its request is still in flight; any late result will be ignored. Waiting before clearing the challenge…" : activeBootstrap ? "Proof cleared. Waiting for the challenge request before cancelling it…" : "Clearing the one-time challenge…");
    controls();
    const operation = (async () => {
      try {
        let csrf = previousCsrf;
        // A bootstrap response can set its challenge cookie after Cancel. Keep
        // starts serialized until it settles, then cancel that exact challenge.
        if (activeBootstrap) csrf = (await activeBootstrap).csrf;
        if (activeProof) await activeProof;
        if (csrf) {
          const reply = await fetch("/identity/cancel", {method: "POST", headers: {Accept: "application/json", "X-Proof-CSRF": csrf}, credentials: "same-origin", cache: "no-store", redirect: "error"});
          if (reply.status !== 401) {
            const value = await responseValue(reply);
            if (!value || value.cancelled !== true || value.authority !== "none") throw new Error("identity_cancel_unconfirmed");
          }
        }
        if (version !== generation) return;
        phase = "idle";
        message("Proof and displayed result cleared. Start a fresh proof when you're ready. No application authority was granted.");
      } catch (error) {
        if (version !== generation) return;
        phase = "idle";
        message("Displayed proof cleared. Challenge cancellation could not be confirmed. Start a fresh proof; no proof response will be retried.");
      } finally { controls(); }
    })();
    cancelInFlight = operation;
    operation.then(() => { if (cancelInFlight === operation) cancelInFlight = null; controls(); });
    return operation;
  }
  start.addEventListener("click", startProof);
  cancel.addEventListener("click", cancelProof);
  addEventListener("pagehide", clearPage);
  addEventListener("pageshow", (event) => { if (event.persisted) clearPage(); });
  if (!trusted()) message("This page is outside the configured identity-proof origin. Identity proof is disabled.");
  controls();
})();
