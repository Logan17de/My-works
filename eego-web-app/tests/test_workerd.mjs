// Optional runtime regression test. Install miniflare or set EEGO_MINIFLARE_MODULE to its module path.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const {Miniflare}=await import(process.env.EEGO_MINIFLARE_MODULE||'miniflare');
const script=await readFile(new URL('../dist/server/index.js',import.meta.url),'utf8');
let status=200, calls=0;
const token='b'.repeat(64);
const expectedCookie=`__Host-eego_session=${token}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=2592000`;
const mf=new Miniflare({modules:true,script,compatibilityDate:'2026-05-01',outboundService:async request=>{
 calls++;
 assert.equal(request.url,'https://jxvabaqswqembehxligi.supabase.co/functions/v1/eego-api/eego');
 assert.equal(request.headers.get('cookie'),'__Host-eego_session=fixture');
 assert.equal(request.headers.get('authorization'),null);
 assert.equal((await request.json()).action,'me');
 return new Response(JSON.stringify({username:'Fixture'}),{status,headers:{'Content-Type':'application/json',...(status===302?{Location:'https://untrusted.example/'}:{'Set-Cookie':`${expectedCookie}, __cf_bm=foreign; Domain=supabase.co; Expires=Sun, 04 Oct 2026 06:35:37 GMT; Secure`})}});
}});
const call=()=>mf.dispatchFetch('https://eego.example/api/eego',{method:'POST',headers:{'Content-Type':'application/json','X-Eego':'1',Origin:'https://eego.example',Cookie:'hosting_session=private; __Host-eego_session=fixture'},body:JSON.stringify({action:'me',data:{}})});
try{
 const result=await call();assert.equal(result.status,200);assert.equal((await result.json()).username,'Fixture');assert.equal(result.headers.get('set-cookie'),expectedCookie);
 status=302;const redirect=await call();assert.equal(redirect.status,502);assert.equal(redirect.headers.get('location'),null);assert.equal(redirect.headers.get('set-cookie'),null);assert.equal(calls,2);
 console.log('PASS: real workerd runtime forwards login/session requests and safely rejects redirects. Upstream mocked.');
}finally{await mf.dispose();}
