// Exercise the actual API against SQLite with D1's prepared statement/batch shape.
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { timingSafeEqual } from 'node:crypto';
import api from '../cloudflare/index.js';
crypto.subtle.timingSafeEqual=timingSafeEqual;
const sql=new DatabaseSync(':memory:');
sql.exec(readFileSync(new URL('../cloudflare/schema.sql',import.meta.url),'utf8'));
function prepared(query,values=[]){return {
 bind(...args){return prepared(query,args);},
 async first(){return sql.prepare(query).get(...values)||null;},
 async all(){return {results:sql.prepare(query).all(...values)};},
 async run(){return {meta:{changes:Number(sql.prepare(query).run(...values).changes)}};}
};}
const db={prepare:prepared,async batch(statements){sql.exec('BEGIN');try{const result=[];for(const s of statements)result.push(await s.run());sql.exec('COMMIT');return result;}catch(e){sql.exec('ROLLBACK');throw e;}}};
const hash=async s=>Buffer.from(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(s))).toString('hex');
const cookie='a'.repeat(64),worker='b'.repeat(64),user='test-user';
const env={DB:db,SITE_TOKEN:'proxy-test',WORKER_TOKEN_HASH:await hash(worker)};
sql.prepare('INSERT INTO users VALUES(?,?,?,?)').run(user,'PrivateTest','unused','unused');
sql.prepare('INSERT INTO sessions VALUES(?,?,?)').run(await hash(cookie),user,Date.now()+86400000);
const seed=JSON.parse(readFileSync(new URL('../data/seed.json',import.meta.url),'utf8'));
for(const i of seed.items)sql.prepare('INSERT INTO items VALUES(?,?,?,?,?,?,?,?)').run(i.id,i.kind,i.level,i.label,i.meaning_ja,i.notes_ja,JSON.stringify(i.examples),i.source);
async function call(action,data={},isWorker=false,authenticated=true){
 if(action==='answer')data={async:true,...data};
 const headers={'Content-Type':'application/json','X-Eego':'1'};
 if(authenticated){if(isWorker)headers.Authorization='Bearer '+worker;else{headers['X-Eego-Proxy']=env.SITE_TOKEN;headers.Cookie='__Host-eego_session='+cookie;}}
 const response=await api.fetch(new Request('https://backend.test/api/eego',{method:'POST',headers,body:JSON.stringify({action,data})}),env);
 return {status:response.status,body:await response.json()};
}
async function ok(action,data={},isWorker=false){const r=await call(action,data,isWorker);assert.equal(r.status,200,JSON.stringify(r.body));return r.body;}
const choice={incremental:true,kind:'vocabulary',level:'B1',count:5,topic:'daily life'};
assert.equal((await call('generate',choice,false,false)).status,403);
assert.equal((await call('worker_claim',{},true,false)).status,401);
const started=await ok('generate',choice);
const initial=await ok('practice',{jobId:started.jobId});
assert.equal(initial.live,true);assert.equal(initial.total,5);assert.deepEqual(initial.questions,[]);
let job=(await ok('worker_claim',{lane:'question'},true)).job;
assert.equal(job.type,'question');assert.equal(job.position,0);
const generated={itemId:job.targets[0].id,sentence:'We hope to ____ this important goal by Friday.',hintJa:'目標を達成すること。'};
assert.equal((await call('worker_complete',{taskId:job.id,lease:job.lease,question:{...generated,answer:'achieve'}},true)).status,400);
await ok('worker_complete',{taskId:job.id,lease:job.lease,question:generated},true);
await ok('worker_complete',{taskId:job.id,lease:job.lease,question:generated},true);
let status=await ok('practice_status',{sessionId:initial.sessionId});
const first=status.questions[0];assert.equal(status.questions.length,1);
assert.equal('answer' in first,false);assert.equal('translationJa' in first,false);assert.equal('explanationJa' in first,false);
assert.equal(sql.prepare('SELECT answer FROM questions WHERE id=?').get(first.id).answer,'');
assert.equal((await ok('worker_claim',{lane:'question'},true)).job,null,'No question 2 before question 1 is read.');
assert.equal((await ok('worker_claim',{lane:'answer'},true)).job,null,'No answer checking before Submit.');
await ok('question_seen',{sessionId:initial.sessionId,questionId:first.id});
await ok('question_seen',{sessionId:initial.sessionId,questionId:first.id});
const next=(await ok('worker_claim',{lane:'question'},true)).job;
assert.equal(next.position,1);
const receipt=await ok('answer',{sessionId:initial.sessionId,questionId:first.id,choice:'banana'});
assert.equal(receipt.pending,true);assert.equal(receipt.state,'queued');
const repeat=await ok('answer',{sessionId:initial.sessionId,questionId:first.id,choice:'different'});
assert.equal(repeat.taskId,receipt.taskId);assert.equal(repeat.choice,'banana');
assert.equal(sql.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,0);
const check=(await ok('worker_claim',{lane:'answer'},true)).job;
assert.equal(check.type,'answer');assert.equal(check.choice,'banana');assert.equal(check.question.sentence,generated.sentence);
assert.equal(check.question.referenceAnswer,'','No answer generated in advance.');
assert.equal(sql.prepare("SELECT COUNT(*) AS n FROM codex_tasks WHERE state='running'").get().n,2,'The next question and answer check can run together.');
const f={correct:false,answer:'achieve',explanationJa:'目標を達成する場合は achieve を使います。',translationJa:'私たちは金曜日までにこの重要な目標を達成したいです。'};
const badLease={taskId:check.id,lease:'x'.repeat(64),feedback:f};
assert.equal((await call('worker_complete',badLease,true)).status,409);
assert.equal(sql.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,0);
await ok('worker_complete',{taskId:check.id,lease:check.lease,feedback:f},true);
await ok('worker_complete',{taskId:check.id,lease:check.lease,feedback:f},true);
assert.equal(sql.prepare('SELECT COUNT(*) AS n FROM attempts').get().n,1);
assert.equal(sql.prepare('SELECT seen FROM mastery').get().seen,1,'Repeated completion does not double-count progress.');
status=await ok('practice_status',{sessionId:initial.sessionId});
assert.equal(status.feedback[first.id].correct,false);assert.equal(status.feedback[first.id].explanationJa,f.explanationJa);
assert.equal((await ok('answer',{sessionId:initial.sessionId,questionId:first.id,choice:'achieve'})).correct,false,'Submitted answers are immutable once checked.');
await ok('worker_complete',{taskId:next.id,lease:next.lease,question:{itemId:next.targets[0].id,sentence:'She hopes to ____ a new skill during the summer.',hintJa:'学ぶことに関する表現。'}},true);
assert.equal((await ok('worker_claim',{lane:'question'},true)).job,null,'No question 3 until question 2 is read.');
status=await ok('practice_status',{sessionId:initial.sessionId});assert.equal(status.questions.length,2);
const second=status.questions[1];
await ok('answer',{sessionId:initial.sessionId,questionId:second.id,choice:'learn'});
const accepted=(await ok('worker_claim',{lane:'answer'},true)).job;
const acceptedFeedback={correct:true,answer:'learn',explanationJa:'この文では learn が自然な表現です。',translationJa:'彼女は夏の間に新しい技能を学びたいと思っています。'};
await ok('worker_fail',{taskId:accepted.id,lease:accepted.lease,reason:'timeout'},true);
const answerRetry=await ok('answer',{sessionId:initial.sessionId,questionId:second.id,choice:'LEARN'});
assert.equal(answerRetry.choice,'LEARN');
const retryCheck=(await ok('worker_claim',{lane:'answer'},true)).job;
assert.equal((await call('worker_complete',{taskId:accepted.id,lease:accepted.lease,feedback:acceptedFeedback},true)).status,409);
await ok('worker_complete',{taskId:retryCheck.id,lease:retryCheck.lease,feedback:acceptedFeedback},true);
assert.equal((await ok('answer',{sessionId:initial.sessionId,questionId:second.id,choice:'LEARN'})).correct,true,'Correctness comes from Codex, rather than literal string equality.');
await ok('question_seen',{sessionId:initial.sessionId,questionId:status.questions[1].id});
const third=(await ok('worker_claim',{lane:'question'},true)).job;assert.equal(third.position,2);
await ok('worker_fail',{taskId:third.id,lease:third.lease,reason:'timeout'},true);
await ok('retry_question',{sessionId:initial.sessionId});
const retried=(await ok('worker_claim',{lane:'question'},true)).job;
assert.equal(retried.id,third.id);assert.notEqual(retried.lease,third.lease);
assert.equal((await call('worker_complete',{taskId:third.id,lease:third.lease,question:generated},true)).status,409);
const outsider=await call('practice_status',{sessionId:'not-owned'});assert.equal(outsider.status,403);
console.log('PASS: question-only generation, read-triggered single look-ahead, concurrent checking after Submit, Japanese feedback, idempotency, scoped sessions and retry leases.');
