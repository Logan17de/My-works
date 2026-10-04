// Only this function possesses the database service key. Browsers get opaque cookies.
const COOKIE = '__Host-eego_session';
const allowed = new Set(['login','logout','connect_worker','password','me','dashboard','catalog','weak','practice','answer','history','report','generate','cancel_job','jobs','export','worker_heartbeat','worker_claim','worker_complete','worker_fail']);
const baseHeaders = {'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store, private','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','Vary':'Cookie'};
function reply(body: unknown, status=200, extra: Record<string,string>={}) { return new Response(JSON.stringify(body),{status,headers:{...baseHeaders,...extra}}); }
async function body(req: Request) {
  const reader=req.body?.getReader(); if(!reader) throw new Error('body');
  let length=0; const chunks:Uint8Array[]=[];
  while(true) { const r=await reader.read(); if(r.done) break; length+=r.value.byteLength; if(length>98304){await reader.cancel(); throw new Error('large');} chunks.push(r.value); }
  const bytes=new Uint8Array(length); let offset=0; for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
  return JSON.parse(new TextDecoder().decode(bytes));
}
export async function handle(req: Request): Promise<Response> {
  try {
    const health = req.method==='GET' && new URL(req.url).pathname.endsWith('/health');
    if(!health && req.method!=='POST') return reply({error:'Method not allowed.'},405,{'Allow':'POST'});
    if(!health && (req.headers.get('x-eego')!=='1' || req.headers.get('sec-fetch-site')==='cross-site')) return reply({error:'Cross-site request refused.'},403);
    if(!health && !req.headers.get('content-type')?.toLowerCase().startsWith('application/json')) return reply({error:'JSON required.'},415);
    const payload=health?{action:'health',data:{}}:await body(req);
    if(!payload || typeof payload!=='object' || Array.isArray(payload) || (!health&&!allowed.has(payload.action)) || (payload.data!=null&&(typeof payload.data!=='object'||Array.isArray(payload.data)))) return reply({error:'Invalid request.'},400);
    const action=payload.action;
    const cookies=Object.fromEntries((req.headers.get('cookie')||'').split(';').map(x=>{const i=x.indexOf('=');return i<0?['','']:[x.slice(0,i).trim(),x.slice(i+1)];}));
    const isWorker=action.startsWith('worker_');
    const token=isWorker?(req.headers.get('authorization')||'').replace(/^Bearer /,''):(cookies[COOKIE]||'');
    if(!health && action!=='login' && !/^[a-f0-9]{64}$/.test(token)) return reply({error:'Please log in.'},401);
    const ip=(req.headers.get('x-real-ip')||req.headers.get('x-forwarded-for')||'unknown').split(',')[0].slice(0,100);
    const serviceKey=Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
    const response=await fetch(`${Deno.env.get('SUPABASE_URL')}/rest/v1/rpc/eego_api`,{method:'POST',headers:{'Content-Type':'application/json','apikey':serviceKey!,'Authorization':`Bearer ${serviceKey}`},body:JSON.stringify({p_action:action,p_data:payload.data||{},p_token:token,p_ip:ip}),signal:AbortSignal.timeout(15000)});
    if(!response.ok){const error=await response.json().catch(()=>({}));console.error('eego_rpc_failure',error.code||response.status);return reply({error:'The request could not be saved. Please try again.'},503);}
    const result=await response.json(); const headers:Record<string,string>={};
    if(action==='login'&&result.token) headers['Set-Cookie']=`${COOKIE}=${result.token}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=2592000`;
    if(action==='logout') headers['Set-Cookie']=`${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0`;
    if(result.status===429) headers['Retry-After']='900';
    return reply(result.error?{error:result.error}:result.data,result.status||200,headers);
  } catch(error) {
    if(error instanceof SyntaxError) return reply({error:'Invalid JSON.'},400);
    if(error instanceof Error&&error.message==='large') return reply({error:'Request too large.'},413);
    console.error('eego_request_failed');return reply({error:'Connection problem. Please try again.'},503);
  }
}
Deno.serve(handle);
