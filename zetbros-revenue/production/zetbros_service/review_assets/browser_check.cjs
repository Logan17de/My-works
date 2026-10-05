/* Local engine QA only. Every browser URL is intercepted; no server/listener. */
"use strict";
const fs = require("node:fs");
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => main(JSON.parse(input)).catch((error) => {
  process.stderr.write(error.stack + "\n");
  process.exitCode = 1;
}));

async function main(fixture) {
  const browser = await chromium.launch({executablePath: fixture.chromium, headless: true,
    args: ["--no-sandbox", "--disable-background-networking", "--disable-sync", "--disable-default-apps", "--no-first-run", "--host-resolver-rules=MAP * ~NOTFOUND"]});
  try {
    const context = await browser.newContext({serviceWorkers: "block", viewport: {width: 1100, height: 900}});
    const requests = [];
    const posts = [];
    let getHandler = null;
    let postHandler = null;
    const proposal = structuredClone(fixture.proposal);
    let ledger = structuredClone(proposal);
    const blocked = [];
    await context.route("**/*", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      requests.push({url: request.url(), headers: await request.allHeaders(), method: request.method()});
      assert.equal(url.origin, fixture.origin, "A request escaped the pinned review origin");
      if (url.pathname === "/review") {
        const html = fs.readFileSync(fixture.assets + "/review.html", "utf8").replace("REVIEW_ORIGIN_PIN", fixture.origin).replace("DELIVERY_NOTICE", "Delivery is disabled. Approval is not evidence of sending or delivery.");
        await route.fulfill({status: 200, contentType: "text/html", body: html,
          headers: {"Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'", "Cache-Control": "no-store"}});
      } else if (url.pathname === "/review/assets/review.js" || url.pathname === "/review/assets/review.css") {
        const name = url.pathname.split("/").pop();
        await route.fulfill({status: 200, contentType: name.endsWith(".js") ? "text/javascript" : "text/css", body: fs.readFileSync(fixture.assets + "/" + name)});
      } else if (url.pathname.startsWith("/review/api/proposals/") && request.method() === "GET") {
        if (getHandler) await getHandler(route, request);
        else await route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(ledger)});
      } else if (url.pathname.endsWith("/decision") && request.method() === "POST") {
        posts.push(request.postDataJSON());
        if (postHandler) await postHandler(route, request);
        else {
          ledger.state = request.postDataJSON().decision === "approve" ? "approved" : "denied";
          await route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(ledger)});
        }
      } else {
        blocked.push(request.url());
        await route.abort("blockedbyclient");
      }
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    async function open() { await page.goto(fixture.origin + "/review"); await page.waitForSelector("#reviewer-token"); }
    async function unlock() {
      await page.fill("#reviewer-token", "fictional_memory_only_token");
      await page.click("#unlock");
      assert.equal(await page.inputValue("#reviewer-token"), "");
    }
    async function load(id = proposal.id) {
      await page.fill("#proposal-id", id);
      await page.click("#load");
      await page.waitForFunction(() => !document.getElementById("preview").hidden);
    }
    async function eligible() {
      await page.check("#confirm-preview");
      assert.equal(await page.isEnabled("#approve"), true);
    }
    await open();
    assert.equal(await page.isEnabled("#load"), false);
    assert.equal(await page.isEnabled("#approve"), false);
    await unlock();
    await load();
    assert.equal(await page.textContent("#reply-body"), proposal.payload.body);
    assert.equal(await page.textContent("#wire-body"), proposal.wire_preview.body);
    assert.deepEqual(JSON.parse(await page.textContent("#action")), proposal.payload);
    assert.deepEqual(JSON.parse(await page.textContent("#wire")), proposal.wire_preview);
    assert.equal(await page.locator("#reply-body script").count(), 0);
    assert.equal(await page.evaluate(() => window.pwned), undefined);
    assert.equal(await page.isEnabled("#approve"), false);
    assert.ok((await page.textContent("#identity")).includes(proposal.digest));
    // Newer input invalidates an already displayed proposal.
    await page.fill("#proposal-id", "00000000-0000-0000-0000-000000000002");
    assert.equal(await page.isVisible("#preview"), false);
    assert.equal(await page.textContent("#reply-body"), "");
    assert.equal(await page.isEnabled("#approve"), false);
    await load();
    await eligible();
    // Repeated click is fenced while the POST is in flight and after success.
    let releasePost;
    postHandler = async (route, request) => {
      await new Promise((resolve) => { releasePost = resolve; });
      ledger.state = "approved";
      await route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(ledger)});
    };
    await page.click("#approve");
    await page.waitForFunction(() => document.getElementById("approve").disabled);
    await page.evaluate(() => document.getElementById("approve").click());
    assert.equal(posts.length, 1);
    assert.deepEqual(posts[0], {digest: proposal.digest, decision: "approve"});
    releasePost();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("Decision recorded"));
    assert.equal(await page.isEnabled("#approve"), false);
    // Logout scrubs token and sensitive preview; no browser-storage artifacts.
    await page.click("#logout");
    assert.equal(await page.isVisible("#preview"), false);
    assert.equal(await page.inputValue("#proposal-id"), "");
    assert.equal(await page.textContent("#action"), "");
    assert.equal(await page.isEnabled("#load"), false);
    assert.deepEqual(await page.evaluate(() => ({local: {...localStorage}, session: {...sessionStorage}, cookie: document.cookie})), {local: {}, session: {}, cookie: ""});
    assert.ok(!page.url().includes("token"));
    // Fresh page, raced GETs: an old response cannot restore stale approval.
    await page.reload(); ledger = structuredClone(proposal); postHandler = null; posts.length = 0;
    await unlock();
    const second = structuredClone(proposal); second.id = "00000000-0000-0000-0000-000000000002"; second.payload.body = "Newer exact preview";
    let oldRoute;
    getHandler = async (route, request) => {
      if (request.url().endsWith(proposal.id)) { oldRoute = route; return; }
      await route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(second)});
    };
    await page.fill("#proposal-id", proposal.id); await page.click("#load");
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("Loading"));
    await page.fill("#proposal-id", second.id); await page.click("#load");
    await page.waitForFunction(() => document.getElementById("reply-body").textContent === "Newer exact preview");
    if (oldRoute) { try { await oldRoute.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(proposal)}); } catch {} }
    assert.equal(await page.textContent("#reply-body"), "Newer exact preview");
    assert.ok((await page.textContent("#identity")).includes(second.id));
    // An interrupted GET followed by Cancel never revives a session/preview.
    await page.click("#cancel");
    assert.equal(await page.isVisible("#preview"), false);
    await unlock();
    let cancelledRoute;
    getHandler = async (route) => { cancelledRoute = route; };
    await page.fill("#proposal-id", proposal.id); await page.click("#load");
    await page.click("#cancel");
    if (cancelledRoute) { try { await cancelledRoute.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(proposal)}); } catch {} }
    assert.equal(await page.isVisible("#preview"), false);
    assert.equal(await page.isEnabled("#load"), false);
    // In-flight POST + Cancel: it cannot be reissued after its result is dropped.
    getHandler = null;
    await unlock(); await load(); await eligible();
    postHandler = async (route) => {
      await new Promise((resolve) => { releasePost = resolve; });
      await route.abort("connectionfailed");
    };
    await page.click("#approve");
    await page.click("#cancel");
    assert.equal(await page.isVisible("#preview"), false);
    assert.equal(await page.isEnabled("#unlock"), false);
    releasePost();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("has settled"));
    await unlock(); await load();
    await page.check("#confirm-preview");
    assert.equal(await page.isEnabled("#approve"), false);
    assert.equal(posts.length, 1);
    assert.ok((await page.textContent("#status")).includes("already attempted"));
    // New page, uncertain POST remains blocked even after a fresh pending GET.
    await page.reload(); await unlock(); await load(); posts.length = 0;
    postHandler = async (route) => { await route.abort("connectionfailed"); };
    await eligible(); await page.click("#deny");
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("outcome is unconfirmed"));
    await page.click("#load");
    await page.waitForFunction(() => !document.getElementById("preview").hidden);
    await page.check("#confirm-preview");
    assert.equal(await page.isEnabled("#deny"), false);
    assert.equal(posts.length, 1);
    // Back/Forward re-entry does not bring back an authorized session.
    await page.goto(fixture.origin + "/review");
    await page.goBack();
    assert.equal(await page.isEnabled("#load"), false);
    assert.equal(await page.isVisible("#preview"), false);
    assert.equal(await page.inputValue("#reviewer-token"), "");
    // Every authorization was a same-origin header, never a URL/cookie.
    for (const request of requests) {
      assert.ok(!request.url.includes("fictional_memory_only_token"));
      assert.equal(request.headers.cookie, undefined);
      if (request.url.includes("/review/api/")) assert.equal(request.headers.authorization, "Bearer fictional_memory_only_token");
      if (request.method === "POST") assert.equal(request.headers.origin, fixture.origin);
    }
    assert.deepEqual(blocked, []);
    assert.deepEqual(errors, []);
    await context.close();
    process.stdout.write("browser-engine checks passed: exact text, no HTML/storage, confirmation, repeats, stale/racing GETs, Cancel/Log out, uncertain POST, Back/Forward\n");
  } finally {
    await browser.close();
  }
}
