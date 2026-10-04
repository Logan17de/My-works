-- Run using an administrative SQL connection. All fixtures are rolled back.
begin;
do $$
declare uid uuid; uid2 uuid; pw text:=encode(extensions.gen_random_bytes(18),'hex'); t text; t2 text; wt text:=encode(extensions.gen_random_bytes(32),'hex'); r jsonb; r2 jsonb; p jsonb; f jsonb; sid uuid; qid uuid; qids uuid[]; jid uuid; lease text; k int; cnt int;
begin
 if has_function_privilege('anon','public.eego_api(text,jsonb,text,text)','execute') then raise exception 'FAIL: anonymous RPC access'; end if;
 if has_schema_privilege('anon','eego','usage') then raise exception 'FAIL: anonymous schema access'; end if;
 insert into eego.users(username,password_hash) values('__eego_test_a',extensions.crypt(pw,extensions.gen_salt('bf',4))) returning id into uid;
 insert into eego.users(username,password_hash) values('__eego_test_b',extensions.crypt(pw,extensions.gen_salt('bf',4))) returning id into uid2;
 insert into eego.items(id,kind,level,label,meaning_ja,notes_ja) values('__test_vocab','vocabulary','B1','borrow','借りる','人から物を借りるときに使います。'),('__test_grammar','grammar','B2','Test pattern','テスト文法','これはテスト専用です。');
 insert into eego.questions(item_id,sentence,options,answer,hint_ja,explanation_ja,translation_ja,fingerprint) values
 ('__test_vocab','May I ____ your umbrella?','["borrow","lend","spill","avoid"]','borrow','借りる','相手から借りるので borrow を使います。','あなたの傘を借りてもいいですか。','__test_q1'),
 ('__test_vocab','Can we ____ a book from the library?','["borrow","lend","spill","avoid"]','borrow','借りる','図書館から借りるので borrow を使います。','図書館で本を借りられますか。','__test_q2');
 r:=public.eego_api('me','{}','','unit-test'); if (r->>'status')::int<>401 then raise exception 'FAIL: auth required'; end if;
 r:=public.eego_api('login',jsonb_build_object('username','__eego_test_a','password',pw),'','test-ip-a'); t:=r->>'token'; if length(t)<>64 then raise exception 'FAIL: login'; end if;
 r:=public.eego_api('login',jsonb_build_object('username','__eego_test_b','password',pw),'','test-ip-b'); t2:=r->>'token';
 r:=public.eego_api('catalog','{"kind":"grammar","level":"B2"}',t,'');
 if exists(select 1 from jsonb_array_elements(r->'data'->'items') x where x->>'kind'<>'grammar') then raise exception 'FAIL: category filtering'; end if;
 p:=public.eego_api('practice','{"itemId":"__test_vocab"}',t,''); sid:=(p->'data'->>'sessionId')::uuid; qid:=(p->'data'->'questions'->0->>'id')::uuid;
 if jsonb_array_length(p->'data'->'questions')<>2 or p->'data'->'questions'->0 ? 'answer' then raise exception 'FAIL: practice or answer leak'; end if;
 r:=public.eego_api('answer',jsonb_build_object('sessionId',sid,'questionId',qid,'choice','lend'),t2,''); if (r->>'status')::int<>403 then raise exception 'FAIL: cross-user session'; end if;
 r:=public.eego_api('answer',jsonb_build_object('sessionId',sid,'questionId',qid,'choice','not-an-option'),t,''); if (r->>'status')::int<>400 then raise exception 'FAIL: answer validation'; end if;
 f:=public.eego_api('answer',jsonb_build_object('sessionId',sid,'questionId',qid,'choice','lend'),t,''); if not (f->'data'->>'weak')::boolean or (f->'data'->>'correct')::boolean then raise exception 'FAIL: weak promotion'; end if;
 r:=public.eego_api('answer',jsonb_build_object('sessionId',sid,'questionId',qid,'choice','borrow'),t,'');
 if r<>f or (select count(*) from eego.attempts where user_id=uid)<>1 then raise exception 'FAIL: answer idempotency'; end if;
 r:=public.eego_api('weak','{}',t,''); if jsonb_array_length(r->'data'->'items')<>1 then raise exception 'FAIL: weak list'; end if;
 qid:=(p->'data'->'questions'->1->>'id')::uuid;
 r:=public.eego_api('answer',jsonb_build_object('sessionId',sid,'questionId',qid,'choice','borrow'),t,'');
 if not (r->'data'->>'weak')::boolean or (r->'data'->>'recovery')::int<>0 then raise exception 'FAIL: immediate retry cleared weak status'; end if;
 for k in 1..3 loop
   update eego.mastery set next_due=now()-interval '1 second' where user_id=uid and item_id='__test_vocab';
   p:=public.eego_api('practice','{"itemId":"__test_vocab"}',t,''); sid:=(p->'data'->>'sessionId')::uuid;
   select (value->>'id')::uuid into qid from jsonb_array_elements(p->'data'->'questions') where (value->>'id')::uuid<>(select last_question from eego.mastery where user_id=uid and item_id='__test_vocab') limit 1;
   r:=public.eego_api('answer',jsonb_build_object('sessionId',sid,'questionId',qid,'choice','borrow'),t,'');
   if (r->'data'->>'recovery')::int<>k then raise exception 'FAIL: spaced recovery step %',k; end if;
 end loop;
 if (r->'data'->>'weak')::boolean then raise exception 'FAIL: graduation'; end if;
 r:=public.eego_api('history','{}',t,''); if jsonb_array_length(r->'data'->'attempts')<>5 then raise exception 'FAIL: history'; end if;
 r:=public.eego_api('history','{}',t2,''); if jsonb_array_length(r->'data'->'attempts')<>0 then raise exception 'FAIL: private history'; end if;
 r:=public.eego_api('export','{}',t,''); if r->'data' ? 'password_hash' or r->'data' ? 'token' or (r->'data'->>'schemaVersion')::int<>1 then raise exception 'FAIL: export'; end if;
 r:=public.eego_api('generate','{"kind":"vocabulary","level":"B1","count":5,"topic":"work","itemId":"__test_vocab"}',t,''); jid:=(r->'data'->>'jobId')::uuid;
 if (r->>'status')::int<>202 then raise exception 'FAIL: generation queue'; end if;
 r:=public.eego_api('generate','{"kind":"vocabulary","level":"B1","count":5,"topic":"work","itemId":"__test_vocab"}',t,''); if (r->>'status')::int<>409 then raise exception 'FAIL: duplicate active generation'; end if;
 r:=public.eego_api('worker_claim','{}',t,''); if (r->>'status')::int<>401 then raise exception 'FAIL: user accessed worker'; end if;
 insert into eego.workers(id,user_id,token_hash) values('__test_worker',uid,encode(extensions.digest(wt,'sha256'),'hex'));
 perform public.eego_api('worker_heartbeat','{"ready":true}',wt,'');
 r:=public.eego_api('dashboard','{}',t2,''); if (r->'data'->>'workerOnline')::boolean then raise exception 'FAIL: cross-user worker status'; end if;
 r:=public.eego_api('worker_claim','{}',wt,''); if r->'data'->'job'->>'id'<>jid::text then raise exception 'FAIL: worker claim'; end if; lease:=r->'data'->'job'->>'lease';
 r:=public.eego_api('worker_claim','{}',wt,''); if r->'data'->'job'<>'null'::jsonb then raise exception 'FAIL: duplicate lease'; end if;
 r:=public.eego_api('worker_complete',jsonb_build_object('jobId',jid,'lease',lease,'questions','[]'::jsonb),wt,''); if (r->>'status')::int<>422 then raise exception 'FAIL: invalid generation accepted'; end if;
 perform public.eego_api('worker_fail',jsonb_build_object('jobId',jid,'lease',lease,'reason','validation'),wt,'');
 if (select state from eego.jobs where id=jid)<>'failed' then raise exception 'FAIL: generation failure'; end if;
 perform public.eego_api('logout','{}',t,''); r:=public.eego_api('me','{}',t,''); if (r->>'status')::int<>401 then raise exception 'FAIL: logout'; end if;
 for k in 1..6 loop r:=public.eego_api('login','{"username":"__eego_test_a","password":"incorrect"}','','test-ip-a'); end loop;
 if (r->>'status')::int<>429 then raise exception 'FAIL: login throttle'; end if;
end $$;
rollback;
select 'PASS: private authentication, filtering, ownership, answer validation, idempotency, weak promotion, delayed recovery, private history, export, generation queue, worker leases, invalid generation rejection, logout and login throttling' as result;
