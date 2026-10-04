const COOKIE='__Host-eego_session', DAY=86400000;
const H={'Cache-Control':'no-store, private','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'};
class ApiError extends Error { constructor(status,message){super(message);this.status=status;} }
const fail=(status,message)=>{throw new ApiError(status,message);};
const enc=new TextEncoder();
const hex=b=>Array.from(new Uint8Array(b),v=>v.toString(16).padStart(2,'0')).join('');
const unhex=s=>Uint8Array.from(s.match(/../g)||[],x=>parseInt(x,16));
const token=()=>hex(crypto.getRandomValues(new Uint8Array(32)));
const hash=async s=>hex(await crypto.subtle.digest('SHA-256',enc.encode(s)));
async function equal(a,b){return crypto.subtle.timingSafeEqual(await crypto.subtle.digest('SHA-256',enc.encode(a||'')),await crypto.subtle.digest('SHA-256',enc.encode(b||'')));}
async function passwordHash(password,salt,pepper){
 const key=await crypto.subtle.importKey('raw',enc.encode(pepper),{name:'HMAC',hash:'SHA-256'},false,['sign']);
 const input=await crypto.subtle.sign('HMAC',key,enc.encode(password));
 const material=await crypto.subtle.importKey('raw',input,'PBKDF2',false,['deriveBits']);
 return hex(await crypto.subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt:unhex(salt),iterations:100000},material,256));
}
const stmt=(db,sql,p=[])=>db.prepare(sql).bind(...p);
const first=(db,sql,p=[])=>stmt(db,sql,p).first();
const all=async(db,sql,p=[])=> (await stmt(db,sql,p).all()).results;
const run=(db,sql,p=[])=>stmt(db,sql,p).run();
const iso=n=>new Date(n).toISOString();
const normalize=s=>s.normalize('NFKC').replace(/[’‘]/g,"'").trim().replace(/\s+/g,' ').toLowerCase();
const text=(v,max=160)=>typeof v==='string'&&v.length<=max?v:'';
const asInt=(v,min,max,def)=>Number.isInteger(v)&&v>=min&&v<=max?v:def;
async function boundedJson(request){
 if(!request.headers.get('content-type')?.startsWith('application/json'))fail(415,'JSON required.');
 const reader=request.body?.getReader();if(!reader)fail(400,'A request is required.');
 const chunks=[];let size=0;
 while(true){const {done,value}=await reader.read();if(done)break;size+=value.length;if(size>98304){await reader.cancel();fail(413,'Request is too large.');}chunks.push(value);}
 const bytes=new Uint8Array(size);let pos=0;for(const part of chunks){bytes.set(part,pos);pos+=part.length;}
 try{return JSON.parse(new TextDecoder().decode(bytes));}catch{fail(400,'Invalid request.');}
}
async function online(db){const row=await first(db,'SELECT ready,last_seen FROM worker_status WHERE id=1');return !!row?.ready&&Date.now()-row.last_seen<90000;}
async function dashboard(db,user){
 const now=Date.now(),jst=new Date(now+9*3600000).toISOString().slice(0,10);
 const start=Date.parse(jst+'T00:00:00+09:00');
 const [stats,study,counts,ready]=await Promise.all([
 first(db,'SELECT COUNT(*) AS totalAnswers, COALESCE(SUM(correct),0) AS correctAnswers, COALESCE(SUM(created_at>=?),0) AS todayAnswers FROM attempts WHERE user_id=?',[start,user.id]),
 first(db,'SELECT COALESCE(SUM(weak),0) AS weakCount, COALESCE(SUM(next_due<=?),0) AS dueCount FROM mastery WHERE user_id=?',[now,user.id]),
 first(db,'SELECT (SELECT COUNT(*) FROM items) AS itemCount,(SELECT COUNT(*) FROM questions q WHERE NOT EXISTS(SELECT 1 FROM reports r WHERE r.user_id=? AND r.question_id=q.id)) AS questionCount',[user.id]),online(db)]);
 return {username:user.username,...stats,...study,...counts,workerOnline:ready};
}
function filter(data,prefix='i'){
 const parts=[],values=[];
 if(['vocabulary','grammar'].includes(data.kind)){parts.push(prefix+'.kind=?');values.push(data.kind);}
 if(['B1','B2','C1','C2'].includes(data.level)){parts.push(prefix+'.level=?');values.push(data.level);}
 return {sql:parts.length?' AND '+parts.join(' AND '):'',values};
}
async function workerAction(db,action,data){
 const now=Date.now();
 if(action==='worker_heartbeat'){await run(db,'UPDATE worker_status SET ready=?,last_seen=? WHERE id=1',[data.ready===true?1:0,now]);return {ok:true};}
 if(action==='worker_claim'){
  await run(db,"UPDATE jobs SET state='failed',error='worker_timeout',lease=NULL WHERE state='running' AND lease_expires<?",[now]);
  const lease=token();
  const job=await first(db,"UPDATE jobs SET state='running',lease=?,lease_expires=? WHERE id=(SELECT id FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1) AND state='queued' RETURNING *",[lease,now+12*60000]);
  if(!job)return {job:null};
  const targets=JSON.parse(job.targets);
  const avoid=await all(db,'SELECT sentence FROM questions WHERE item_id IN (SELECT value FROM json_each(?)) ORDER BY rowid DESC LIMIT 100',[JSON.stringify(targets.map(t=>t.id))]);
  return {job:{id:job.id,count:job.count,topic:job.topic,targets,avoidSentences:avoid.map(x=>x.sentence),lease}};
 }
 if(!['worker_complete','worker_fail'].includes(action))fail(403,'Worker action not allowed.');
 const job=await first(db,'SELECT * FROM jobs WHERE id=?',[text(data.jobId)]);
 if(!job)fail(404,'Batch not found.');
 if(job.state==='completed'&&await equal(job.lease,text(data.lease)))return {saved:job.question_count};
 if(job.state!=='running'||job.lease_expires<now||!(await equal(job.lease,text(data.lease))))fail(409,'This batch lease has expired.');
 if(action==='worker_fail'){
  const reason=['timeout','validation','auth_required','quota','worker_error'].includes(data.reason)?data.reason:'worker_error';
  await run(db,"UPDATE jobs SET state='failed',error=? WHERE id=? AND lease=? AND state='running'",[reason,job.id,job.lease]);return {ok:true};
 }
 const qs=data.questions,targets=new Set(JSON.parse(job.targets).map(t=>t.id)),seen=new Set();
 if(!Array.isArray(qs)||qs.length!==job.count)fail(400,'Incorrect question count.');
 const statements=[];
 for(const q of qs){
  if(!q||!targets.has(q.itemId)||typeof q.sentence!=='string'||q.sentence.length<15||q.sentence.length>600||(q.sentence.match(/____/g)||[]).length!==1)fail(400,'Invalid question.');
  if(!Array.isArray(q.options)||q.options.length!==4||q.options.some(v=>typeof v!=='string'||!v.trim()||v.length>160)||new Set(q.options.map(normalize)).size!==4||!q.options.includes(q.answer))fail(400,'Invalid answer metadata.');
  for(const [field,max] of [['hintJa',300],['explanationJa',1500],['translationJa',1000]])if(typeof q[field]!=='string'||!q[field].trim()||q[field].length>max||/[<>]/.test(q[field]))fail(400,'Japanese support is required.');
  if(!/[ぁ-んァ-ヶ一-龯]/.test(q.explanationJa)||!/[ぁ-んァ-ヶ一-龯]/.test(q.translationJa))fail(400,'Japanese explanation and translation are required.');
  if(/[<>]/.test(q.sentence)||/https?:\/\//.test(q.sentence))fail(400,'Invalid question text.');
  const fingerprint=await hash(normalize(q.sentence));if(seen.has(fingerprint))fail(400,'Duplicate question.');seen.add(fingerprint);
  statements.push(stmt(db,"INSERT INTO questions(id,item_id,sentence,options,answer,hint_ja,explanation_ja,translation_ja,source,fingerprint,job_id) SELECT ?,?,?,?,?,?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM jobs WHERE id=? AND lease=? AND state='running' AND lease_expires>=?)",[crypto.randomUUID(),q.itemId,q.sentence,JSON.stringify(q.options),q.answer,q.hintJa,q.explanationJa,q.translationJa,'Generated with Codex',fingerprint,job.id,job.id,job.lease,now]));
 }
 statements.push(stmt(db,"UPDATE jobs SET state='completed',question_count=? WHERE id=? AND lease=? AND state='running' AND lease_expires>=?",[qs.length,job.id,job.lease,now]));
 const saved=await db.batch(statements);if(saved.at(-1).meta.changes!==1)fail(409,'This batch lease has expired.');
 return {saved:qs.length};
}
async function appAction(db,env,request,action,data,headers){
 const now=Date.now();
 const cookie=(request.headers.get('cookie')||'').match(/(?:^|;\s*)__Host-eego_session=([a-f0-9]{64})(?:;|$)/)?.[1];
 if(action==='login'){
  const limitKey=await hash(request.headers.get('x-eego-client-ip')||'unknown');
  await run(db,'INSERT INTO login_limits(key,attempts,window) VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET attempts=CASE WHEN window<? THEN 1 ELSE attempts+1 END,window=CASE WHEN window<? THEN excluded.window ELSE window END',[limitKey,now,now-10*60000,now-10*60000]);
  const limit=await first(db,'SELECT attempts FROM login_limits WHERE key=?',[limitKey]);if(limit.attempts>8)fail(429,'Too many attempts. Please wait 10 minutes.');
  const user=await first(db,'SELECT * FROM users WHERE username=? COLLATE NOCASE',[text(data.username,80).trim()]);
  const salt=user?.salt||'00000000000000000000000000000000';
  const digest=await passwordHash(text(data.password,128),salt,env.AUTH_PEPPER);
  if(!user||!(await equal(digest,user.password_hash)))fail(401,'Username or password is incorrect.');
  const raw=token();
  await db.batch([stmt(db,'INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)',[await hash(raw),user.id,now+30*DAY]),stmt(db,'DELETE FROM login_limits WHERE key=?',[limitKey]),stmt(db,'DELETE FROM sessions WHERE expires_at<?',[now])]);
  headers.set('Set-Cookie',`${COOKIE}=${raw}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=2592000`);
  return {username:user.username};
 }
 const sessionHash=cookie?await hash(cookie):'';
 const user=await first(db,'SELECT u.id,u.username FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.token_hash=? AND s.expires_at>?',[sessionHash,now]);
 if(!user)fail(401,'Please sign in.');
 if(action==='logout'){await run(db,'DELETE FROM sessions WHERE token_hash=?',[sessionHash]);headers.set('Set-Cookie',`${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0`);return {ok:true};}
 if(action==='me'||action==='dashboard')return dashboard(db,user);
 if(action==='catalog'||action==='weak'){
  const f=filter(data),search=text(data.search,80).trim();
  const rows=await all(db,`SELECT i.*,COALESCE(m.seen,0) AS seen,COALESCE(m.correct,0) AS correct,COALESCE(m.weak,0) AS weak,COALESCE(m.recovery,0) AS recovery,m.next_due,(SELECT COUNT(*) FROM questions q WHERE q.item_id=i.id AND NOT EXISTS(SELECT 1 FROM reports r WHERE r.user_id=? AND r.question_id=q.id)) AS question_count FROM items i LEFT JOIN mastery m ON m.item_id=i.id AND m.user_id=? WHERE 1=1 ${f.sql} ${action==='weak'?'AND m.weak=1':''} AND (i.label LIKE ? OR i.meaning_ja LIKE ?) ORDER BY i.level,i.label LIMIT 100 OFFSET ?`,[user.id,user.id,...f.values,'%'+search+'%','%'+search+'%',asInt(data.offset,0,10000,0)]);
  return {items:rows.map(r=>({...r,weak:!!r.weak,examples:JSON.parse(r.examples),next_due:r.next_due?iso(r.next_due):null}))};
 }
 if(action==='practice'){
  const f=filter(data),conditions=[],params=[user.id,user.id,...f.values];
  if(data.itemId){conditions.push('q.item_id=?');params.push(text(data.itemId));}
  if(data.jobId){conditions.push('q.job_id=?');params.push(text(data.jobId));}
  if(data.mode==='weak')conditions.push('m.weak=1');
  if(data.mode==='due'){conditions.push('m.next_due<=?');params.push(now);}
  const qs=await all(db,`SELECT q.id,q.sentence,q.hint_ja AS hintJa,q.translation_ja AS translationJa,q.source,i.kind,i.level FROM questions q JOIN items i ON i.id=q.item_id LEFT JOIN mastery m ON m.item_id=i.id AND m.user_id=? WHERE NOT EXISTS(SELECT 1 FROM reports r WHERE r.user_id=? AND r.question_id=q.id) ${f.sql} ${conditions.length?'AND '+conditions.join(' AND '):''} ORDER BY RANDOM() LIMIT ${data.jobId?20:10}`,params);
  const id=crypto.randomUUID();if(qs.length)await run(db,'INSERT INTO practice(id,user_id,questions,expires_at) VALUES(?,?,?,?)',[id,user.id,JSON.stringify(qs.map(q=>q.id)),now+DAY]);
  return {sessionId:id,questions:qs,message:qs.length?'':'No questions for this selection yet.'};
 }
 if(action==='answer'){
  const session=await first(db,'SELECT * FROM practice WHERE id=? AND user_id=? AND expires_at>?',[text(data.sessionId),user.id,now]);
  if(!session||!JSON.parse(session.questions).includes(data.questionId))fail(403,'This question is not in your session.');
  const existing=await first(db,'SELECT feedback FROM attempts WHERE session_id=? AND question_id=?',[session.id,text(data.questionId)]);if(existing)return JSON.parse(existing.feedback);
  const q=await first(db,'SELECT q.*,i.label,i.meaning_ja FROM questions q JOIN items i ON i.id=q.item_id WHERE q.id=?',[data.questionId]);
  const choice=text(data.choice);if(!choice.trim())fail(400,'Type your answer first.');
  const correct=normalize(choice)===normalize(q.answer),unsure=data.unsure===true;
  const prev=await first(db,'SELECT * FROM mastery WHERE user_id=? AND item_id=?',[user.id,q.item_id]);
  let weak=!!prev?.weak,recovery=prev?.recovery||0,due=prev?.next_due||now+DAY;
  if(!correct||unsure){weak=true;recovery=0;due=now+10*60000;}
  else if(weak&&due<=now&&prev.last_question!==q.id){recovery=Math.min(3,recovery+1);weak=recovery<3;due=now+([1,3,7][recovery-1])*DAY;}
  else if(!weak)due=now+7*DAY;
  const feedback={correct,unsure,choice,answer:q.answer,sentence:q.sentence,itemId:q.item_id,label:q.label,meaningJa:q.meaning_ja,explanationJa:q.explanation_ja,translationJa:q.translation_ja,weak,recovery,nextDue:iso(due)};
  const id=crypto.randomUUID();
  await db.batch([
   stmt(db,'INSERT OR IGNORE INTO attempts(id,user_id,session_id,question_id,item_id,choice,correct,unsure,created_at,feedback) VALUES(?,?,?,?,?,?,?,?,?,?)',[id,user.id,session.id,q.id,q.item_id,choice,Number(correct),Number(unsure),now,JSON.stringify(feedback)]),
   stmt(db,'INSERT INTO mastery(user_id,item_id,seen,correct,weak,recovery,next_due,last_question) SELECT ?,?,1,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM attempts WHERE id=?) ON CONFLICT(user_id,item_id) DO UPDATE SET seen=mastery.seen+1,correct=mastery.correct+excluded.correct,weak=excluded.weak,recovery=excluded.recovery,next_due=excluded.next_due,last_question=excluded.last_question',[user.id,q.item_id,Number(correct),Number(weak),recovery,due,q.id,id])]);
  return JSON.parse((await first(db,'SELECT feedback FROM attempts WHERE session_id=? AND question_id=?',[session.id,q.id])).feedback);
 }
 if(action==='history'){
  const rows=await all(db,`SELECT a.*,q.sentence,q.answer,q.explanation_ja,q.translation_ja,i.label,i.kind,i.level FROM attempts a JOIN questions q ON q.id=a.question_id JOIN items i ON i.id=a.item_id WHERE a.user_id=? ${data.mistakes?'AND (a.correct=0 OR a.unsure=1)':''} ORDER BY a.created_at DESC LIMIT 30 OFFSET ?`,[user.id,asInt(data.offset,0,100000,0)]);
  return {attempts:rows.map(r=>({...r,feedback:JSON.parse(r.feedback),created_at:iso(r.created_at),correct:!!r.correct,unsure:!!r.unsure}))};
 }
 if(action==='report'){await run(db,'INSERT OR IGNORE INTO reports(user_id,question_id) SELECT ?,id FROM questions WHERE id=?',[user.id,text(data.questionId)]);return {ok:true};}
 if(action==='generate'){
  if(!['vocabulary','grammar'].includes(data.kind)||!['B1','B2','C1','C2'].includes(data.level)||![5,10,20].includes(data.count)||!['mixed','daily life','work','travel','study'].includes(data.topic))fail(400,'Choose a category, level, and batch size.');
  const pending=await first(db,"SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND state IN ('queued','running')",[user.id]);if(pending.n>=2)fail(429,'Please finish or cancel a pending batch first.');
  const recent=await first(db,'SELECT COALESCE(SUM(count),0) AS n FROM jobs WHERE user_id=? AND created_at>?',[user.id,now-DAY]);if(recent.n+data.count>150)fail(429,'Today’s generation limit is reached. Saved lessons are ready to practise.');
  const targets=await all(db,`SELECT i.* FROM items i LEFT JOIN mastery m ON m.item_id=i.id AND m.user_id=? WHERE i.kind=? AND i.level=? ${data.weakOnly?'AND m.weak=1':''} ORDER BY RANDOM() LIMIT 5`,[user.id,data.kind,data.level]);
  if(!targets.length)fail(400,'No matching learning targets. Try another level or turn off weak items only.');
  const id=crypto.randomUUID();
  const inserted=await run(db,"INSERT INTO jobs(id,user_id,kind,level,count,topic,targets,created_at) SELECT ?,?,?,?,?,?,?,? WHERE (SELECT COUNT(*) FROM jobs WHERE user_id=? AND state IN ('queued','running'))<2 AND (SELECT COALESCE(SUM(count),0) FROM jobs WHERE user_id=? AND created_at>?)<=?",[id,user.id,data.kind,data.level,data.count,data.topic,JSON.stringify(targets.map(t=>({...t,examples:JSON.parse(t.examples)}))),now,user.id,user.id,now-DAY,150-data.count]);
  if(inserted.meta.changes!==1)fail(429,'Please try later or finish a pending batch first.');
  return {jobId:id,workerOnline:await online(db)};
 }
 if(action==='jobs'){const rows=await all(db,'SELECT id,kind,level,count,topic,state,created_at,error,question_count FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 30',[user.id]);return {jobs:rows.map(r=>({...r,created_at:iso(r.created_at)}))};}
 if(action==='cancel_job'){await run(db,"UPDATE jobs SET state='cancelled',error='cancelled' WHERE id=? AND user_id=? AND state='queued'",[text(data.jobId),user.id]);return {ok:true};}
 if(action==='export')return {username:user.username,exportedAt:iso(now),attempts:await all(db,'SELECT question_id,item_id,choice,correct,unsure,created_at,feedback FROM attempts WHERE user_id=? ORDER BY created_at',[user.id]),mastery:await all(db,'SELECT item_id,seen,correct,weak,recovery,next_due FROM mastery WHERE user_id=?',[user.id])};
 if(action==='password'){
  const account=await first(db,'SELECT * FROM users WHERE id=?',[user.id]);
  if(!(await equal(await passwordHash(text(data.currentPassword,128),account.salt,env.AUTH_PEPPER),account.password_hash)))fail(401,'Current password is incorrect.');
  if(typeof data.newPassword!=='string'||data.newPassword.length<10||data.newPassword.length>128)fail(400,'Use at least 10 characters.');
  const salt=token(),digest=await passwordHash(data.newPassword,salt,env.AUTH_PEPPER);
  await db.batch([stmt(db,'UPDATE users SET password_hash=?,salt=? WHERE id=?',[digest,salt,user.id]),stmt(db,'DELETE FROM sessions WHERE user_id=? AND token_hash<>?',[user.id,sessionHash])]);return {ok:true};
 }
 fail(400,'Unknown action.');
}
export default {async fetch(request,env){
 const headers=new Headers(H),url=new URL(request.url);
 if(url.pathname==='/health'&&request.method==='GET')return Response.json({ok:true,app:'Eego',database:'Cloudflare D1'},{headers});
 if(url.pathname!=='/api/eego')return new Response('Not found',{status:404,headers});
 if(request.method!=='POST')return Response.json({error:'Method not allowed.'},{status:405,headers});
 try{
  const body=await boundedJson(request);
  if(!body||typeof body!=='object'||Array.isArray(body))fail(400,'Invalid request.');
  const action=text(body.action,40),data=body.data&&typeof body.data==='object'&&!Array.isArray(body.data)?body.data:{};
  let result;
  if(action.startsWith('worker_')){
   const supplied=request.headers.get('authorization')?.replace(/^Bearer /,'')||'';
   if(!(await equal(await hash(supplied),env.WORKER_TOKEN_HASH)))fail(401,'Worker authentication failed.');
   result=await workerAction(env.DB,action,data);
  }else{
   if(!(await equal(request.headers.get('x-eego-proxy'),env.SITE_TOKEN)))fail(403,'Use the Eego app to sign in.');
   result=await appAction(env.DB,env,request,action,data,headers);
  }
  return Response.json(result,{headers});
 }catch(error){
  if(error instanceof ApiError)return Response.json({error:error.message},{status:error.status,headers});
  console.error(JSON.stringify({event:'eego_api_failed',type:error?.name||'Error'}));
  return Response.json({error:'Eego is temporarily unavailable. Please try again.'},{status:503,headers});
 }
}};
