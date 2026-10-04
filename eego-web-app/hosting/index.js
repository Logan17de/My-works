const API = 'https://eego-api.zetbros.workers.dev/api/eego';
const COOKIE = '__Host-eego_session';
const headers = {
  'Cache-Control': 'no-store, private',
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'X-Frame-Options': 'DENY',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
};
const error = (message, status) => Response.json({error:message},{status,headers});
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if(url.pathname === '/api/eego') {
      if(request.method !== 'POST') return error('Method not allowed.',405);
      if(request.headers.get('origin') && request.headers.get('origin') !== url.origin) return error('Cross-site request refused.',403);
      if(request.headers.get('sec-fetch-site') === 'cross-site' || request.headers.get('x-eego') !== '1') return error('Cross-site request refused.',403);
      if(!request.headers.get('content-type')?.toLowerCase().startsWith('application/json')) return error('JSON required.',415);
      if(!env?.EEGO_PROXY_TOKEN) return error('Eego is temporarily unavailable.',503);
      try {
        const reader=request.body?.getReader();
        if(!reader) return error('Request body required.',400);
        let size=0; const parts=[];
        while(true){const part=await reader.read();if(part.done)break;size+=part.value.length;if(size>98304){await reader.cancel();return error('Request too large.',413);}parts.push(part.value);}
        const bytes=new Uint8Array(size);let pos=0;for(const p of parts){bytes.set(p,pos);pos+=p.length;}
        const cookie=(request.headers.get('cookie')||'').split(';').map(v=>v.trim()).find(v=>v.startsWith(COOKIE+'='));
        const upstreamHeaders={'Content-Type':'application/json','X-Eego':'1','User-Agent':'Eego-Sites-Proxy/2.0','X-Eego-Proxy':env.EEGO_PROXY_TOKEN,'X-Eego-Client-IP':request.headers.get('cf-connecting-ip')||'unknown'};
        // Only the Eego session travels upstream. Never forward the hosting session.
        if(cookie) upstreamHeaders.Cookie=cookie;
        // Workerd only supports follow/manual; reject redirects explicitly so credentials never follow one.
        const response=await fetch(API,{method:'POST',headers:upstreamHeaders,body:bytes,redirect:'manual',signal:AbortSignal.timeout(20000)});
        if(response.status>=300 && response.status<400){await response.body?.cancel();return error('The authentication server redirected unexpectedly. Please try again.',502);}
        const out=new Headers(headers);out.set('Content-Type','application/json; charset=utf-8');
        // Fetch may combine the backend session and Cloudflare cookies into one header.
        // Rebuild only our session cookie: foreign Domain/Expires attributes would make
        // browsers reject the __Host- cookie and immediately lose a successful login.
        const session=response.headers.get('set-cookie')?.match(/(?:^|,\s*)__Host-eego_session=([a-f0-9]{64}|)(?=;|$)/);
        if(session) out.set('Set-Cookie',`${COOKIE}=${session[1]}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=${session[1]?2592000:0}`);
        if(response.headers.has('retry-after'))out.set('Retry-After',response.headers.get('retry-after'));
        return new Response(response.body,{status:response.status,headers:out});
      }catch(e){console.error('eego_proxy_failed',e instanceof Error?e.name:'UnknownError');return error('Cannot reach Eego. Please try again.',503);}
    }
    if(request.method!=='GET' && request.method!=='HEAD') return error('Method not allowed.',405);
    const path=url.pathname==='/'?'/index.html':url.pathname;
    const asset=ASSETS[path];
    if(!asset)return new Response('Page not found.',{status:404,headers});
    return new Response(request.method==='HEAD'?null:asset.body,{headers:{...headers,'Content-Type':asset.type}});
  }
};
