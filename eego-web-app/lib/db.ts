const url=()=>{if(!process.env.SUPABASE_URL)throw new Error("SUPABASE_URL missing");return process.env.SUPABASE_URL};
const key=()=>{if(!process.env.SUPABASE_PUBLISHABLE_KEY)throw new Error("SUPABASE_PUBLISHABLE_KEY missing");return process.env.SUPABASE_PUBLISHABLE_KEY};
const gate=()=>{if(!process.env.EEGO_DB_GATE)throw new Error("EEGO_DB_GATE missing");return process.env.EEGO_DB_GATE};
export async function db<T=unknown>(path:string,init:RequestInit={}):Promise<T>{
 const res=await fetch(`${url()}/rest/v1/${path}`,{...init,cache:"no-store",headers:{apikey:key(),Authorization:`Bearer ${key()}`,"x-eego-key":gate(),"Content-Type":"application/json",...(init.headers||{})}});
 if(!res.ok)throw new Error(`Database ${res.status}: ${await res.text()}`);
 if(res.status===204)return undefined as T;
 return res.json() as Promise<T>;
}
export function isoAfterDays(days:number){return new Date(Date.now()+days*86400000).toISOString()}
