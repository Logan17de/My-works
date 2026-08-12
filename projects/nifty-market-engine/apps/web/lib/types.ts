export type SignalPayload = {
  timestamp: string;
  event: "breakout" | "reversal" | "uncertain" | "no_level";
  direction: "bullish" | "bearish" | "flat";
  confidence: number;
  combined_direction_score: number;
  cash: { pressure:number; breadth:number; participation:number; signed_volume_acceleration:number; score:number; advancers:number; decliners:number };
  futures: { price_direction:number; volume_activity:number; oi_confirmation:number; basis_change:number; score:number };
  level: { event:string; event_score:number; breakout_score:number; reversal_score:number; distance_bps:number; level_name:string|null };
  contract: { score:number; reason:string; contract:null|{ trading_symbol:string; strike:number; option_type:"CE"|"PE"; ltp:number } };
  risk: { allowed:boolean; quantity:number; reason:string };
  reasons: string[];
};
