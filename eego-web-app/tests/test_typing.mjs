// DOM interaction tests with a mocked API; no browser or production study records.
// Install linkedom separately or set EEGO_DOM_MODULE to its ESM entrypoint.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const {parseHTML}=await import(process.env.EEGO_DOM_MODULE||'linkedom');
const {window,document}=parseHTML(await readFile(new URL('../web/index.html',import.meta.url),'utf8'));
const seed=JSON.parse(await readFile(new URL('../data/seed.json',import.meta.url),'utf8'));
const borrow=seed.questions.find(q=>q.answer==='borrow');
const storage=new Map(),requests=[];
let fail=false,answerCount=0;
const stats={username:'Fixture',totalAnswers:0,correctAnswers:0,todayAnswers:0,weakCount:0,dueCount:0,learnedCount:0,itemCount:1,questionCount:2,workerOnline:false};
const questions=[{id:'q1',kind:'vocabulary',level:'B1',sentence:borrow.sentence,hintJa:'借りる',source:'Test fixture'},{id:'q2',kind:'grammar',level:'B2',sentence:'I ____ known if you had told me.',translationJa:'教えてくれていたら知っていたでしょう。',hintJa:'過去の仮定',source:'Test fixture'}];
const fetch=async(_url,options)=>{
 const {action,data}=JSON.parse(options.body);requests.push({action,data});
 let body=stats,status=200;
 if(action==='practice'){assert.equal(data.format,'typing');body={sessionId:'fixture-session',questions};}
 if(action==='answer'){
  answerCount++;
  if(fail){fail=false;body={error:'Connection failed. Please retry.'};status=503;}
  else{const answer=data.questionId==='q1'?'borrow':'would have';const correct=data.choice.trim().toLowerCase()===answer;body={correct,answer,choice:data.choice,unsure:data.unsure,weak:!correct||data.unsure,recovery:0,nextDue:'2026-10-05T00:00:00Z',label:answer,meaningJa:'意味',explanationJa:'日本語の説明',translationJa:'例文の訳'};}
 }
 return new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
};
window.scrollTo=()=>{};
const context=vm.createContext({window,document,console,fetch,Response,AbortSignal,Intl,URL,Blob,FormData,
 sessionStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
 setTimeout:()=>0,clearTimeout:()=>{},setInterval:()=>0,clearInterval:()=>{}});
vm.runInContext(await readFile(new URL('../web/app.js',import.meta.url),'utf8'),context);
const bundled=vm.runInContext('lessonTranslations',context);
assert.equal(Object.keys(bundled).length,seed.questions.length);
for(const q of seed.questions)assert.equal(bundled[q.sentence],q.translation_ja,'All starter translations must match the saved bank');
const wait=async check=>{for(let i=0;i<30;i++){if(check())return;await new Promise(resolve=>setImmediate(resolve));}assert.ok(check(),'Expected UI state was not reached');};
const event=(type)=>new window.Event(type,{bubbles:true,cancelable:true});
const click=selector=>{const el=document.querySelector(selector);assert.ok(el,selector);el.dispatchEvent(event('click'));};
const input=value=>{const el=document.querySelector('#typed-answer');el.value=value;el.dispatchEvent(event('input'));};
const submit=()=>document.querySelector('#answer-form').dispatchEvent(event('submit'));
await wait(()=>document.querySelector('[data-action="start"]'));
click('[data-action="start"]');await wait(()=>document.querySelector('#typed-answer'));
assert.equal(document.querySelectorAll('.sentence input').length,1);
assert.equal(document.querySelectorAll('.option').length,0);
assert.equal(document.querySelector('#question-translation p').textContent,borrow.translation_ja);
assert.equal(document.querySelector('#question-translation').getAttribute('lang'),'ja');
assert.equal(document.querySelector('.feedback'),null);
assert.equal(JSON.parse(storage.get('eego_active_v1')).hintUsed,false);
assert.equal(JSON.parse(storage.get('eego_active_v1')).unsure,false,'Reading the translation is not a guess');
assert.equal(document.querySelector('#typed-answer').getAttribute('autocomplete'),'off');
assert.equal(document.querySelector('#typed-answer').getAttribute('spellcheck'),'false');
input('   ');assert.ok(document.querySelector('#check-answer').disabled);submit();assert.equal(answerCount,0);
input('borow');assert.equal(document.querySelector('#check-answer').disabled,false);
click('[data-nav="home"]');await wait(()=>document.querySelector('[data-action="resume"]'));click('[data-action="resume"]');
assert.equal(document.querySelector('#typed-answer').value,'borow');
assert.equal(document.querySelector('#question-translation p').textContent,borrow.translation_ja);
assert.equal(requests.filter(r=>r.action==='practice').length,1,'Old saved questions get their translations without restarting a lesson');
click('[data-action="hint"]');assert.equal(document.querySelector('#typed-answer').value,'borow');
assert.ok(JSON.parse(storage.get('eego_active_v1')).unsure);
const ime=event('keydown');Object.defineProperties(ime,{key:{value:'Enter'},isComposing:{value:true}});document.querySelector('#typed-answer').dispatchEvent(ime);assert.ok(ime.defaultPrevented);
fail=true;submit();await wait(()=>!document.querySelector('#typed-answer').disabled);
assert.equal(document.querySelector('#typed-answer').value,'borow');assert.equal(document.querySelector('.feedback'),null);
assert.match(document.querySelector('#toast').textContent,/Connection failed/);
submit();submit();await wait(()=>document.querySelector('.feedback'));
assert.equal(answerCount,2,'A retry plus one successful submission; double submit must be ignored');
assert.ok(document.querySelector('.gap-input.wrong'));assert.match(document.querySelector('.your-answer').textContent,/borow/);
assert.equal(JSON.parse(storage.get('eego_active_v1')).index,0);
click('[data-action="next"]');assert.equal(document.querySelector('#typed-answer').value,'');assert.ok(document.querySelector('#check-answer').disabled);
assert.equal(document.querySelector('#question-translation p').textContent,questions[1].translationJa);
input('  WOULD HAVE  ');submit();await wait(()=>document.querySelector('.feedback'));
assert.ok(document.querySelector('.gap-input.correct'));assert.equal(requests.filter(r=>r.action==='answer').at(-1).data.choice,'  WOULD HAVE  ');
click('[data-action="next"]');assert.match(document.querySelector('.summary-score').textContent,/1.*2/);assert.equal(storage.has('eego_active_v1'),false);
console.log('PASS: all 80 saved translations, Japanese before answering, resumed drafts, supplied translations, typed blanks, retries, grading and completion. API mocked; browser layout not tested.');
