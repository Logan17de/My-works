"use client";

import { useEffect, useState } from "react";
import { browserSupabase } from "@/lib/supabase";
import type { SignalPayload } from "@/lib/types";

function pct(value: number) { return `${(value * 100).toFixed(0)}%`; }

export default function SignalDashboard() {
  const [signal, setSignal] = useState<SignalPayload | null>(null);
  const [status, setStatus] = useState("waiting for Supabase configuration");

  useEffect(() => {
    const supabase = browserSupabase();
    if (!supabase) return;
    let mounted = true;
    async function loadLatest() {
      const { data, error } = await supabase.from("signals").select("payload")
        .order("observed_at", { ascending: false }).limit(1).maybeSingle();
      if (!mounted) return;
      if (error) setStatus(error.message);
      else if (data?.payload) { setSignal(data.payload as SignalPayload); setStatus("live"); }
      else setStatus("connected — no signals yet");
    }
    void loadLatest();
    const channel = supabase.channel("signals-dashboard").on(
      "postgres_changes", { event: "INSERT", schema: "public", table: "signals" },
      (payload) => {
        const row = payload.new as { payload?: SignalPayload };
        if (row.payload) { setSignal(row.payload); setStatus("live"); }
      },
    ).subscribe();
    return () => { mounted = false; void supabase.removeChannel(channel); };
  }, []);

  return (
    <main className="shell">
      <header><p className="eyebrow">NIFTY MARKET ENGINE</p><h1>Level-event monitor</h1><p className="muted">{status}</p></header>
      {!signal ? <section className="card"><p>No signal payload yet.</p></section> : <>
        <section className="hero card">
          <div><span className="label">STATE</span><strong>{signal.event.toUpperCase()} · {signal.direction.toUpperCase()}</strong></div>
          <div><span className="label">CONFIDENCE</span><strong>{pct(signal.confidence)}</strong></div>
          <div><span className="label">LEVEL</span><strong>{signal.level.level_name ?? "—"}</strong></div>
          <div><span className="label">RISK GATE</span><strong>{signal.risk.allowed ? "ALLOW" : "BLOCK"}</strong></div>
        </section>
        <section className="grid">
          <article className="card"><span className="label">CASH PARTICIPATION</span><h2>{signal.cash.score.toFixed(3)}</h2><p>Pressure {signal.cash.pressure.toFixed(3)} · breadth {signal.cash.advancers}/{signal.cash.decliners}</p><p>Participation {pct(signal.cash.participation)}</p></article>
          <article className="card"><span className="label">FUTURES</span><h2>{signal.futures.score.toFixed(3)}</h2><p>OI confirmation {signal.futures.oi_confirmation.toFixed(3)}</p><p>Basis change {signal.futures.basis_change.toFixed(3)}</p></article>
          <article className="card"><span className="label">LEVEL CLASSIFIER</span><h2>{signal.level.event_score.toFixed(3)}</h2><p>Breakout {signal.level.breakout_score.toFixed(3)}</p><p>Reversal {signal.level.reversal_score.toFixed(3)}</p></article>
          <article className="card"><span className="label">OPTION</span><h2>{signal.contract.contract?.trading_symbol ?? "NO CONTRACT"}</h2><p>Selection score {signal.contract.score.toFixed(3)}</p><p>Quantity {signal.risk.quantity}</p></article>
        </section>
        <section className="card"><span className="label">WHY</span><ul>{signal.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></section>
      </>}
    </main>
  );
}
