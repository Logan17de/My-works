create or replace function public.eego_api(p_action text, p_data jsonb default '{}'::jsonb, p_token text default '', p_ip text default '') returns jsonb
language plpgsql security definer set search_path = '' as $$
#variable_conflict use_column
declare
 u eego.users; w eego.workers; j eego.jobs; q eego.questions; m eego.mastery; s eego.practice;
 uid uuid; token text; keyip text; keyuser text; n int; total int; ids uuid[]; targets text[]; choice text; ok boolean; unsure boolean;
 v jsonb; outjson jsonb; arr jsonb; fb jsonb; mode text; v_kind text; lvl text; jobuuid uuid; requested int; step int; due timestamptz; isweak boolean;
begin
 if p_action='health' then return jsonb_build_object('status',200,'data',jsonb_build_object('app','Eego','version','1.0.0','database',true)); end if;
 if p_action='login' then
   if length(coalesce(p_data->>'username',''))>80 or length(coalesce(p_data->>'password',''))>128 then return '{"status":400,"error":"Invalid login."}'; end if;
   keyip := 'ip:'||encode(extensions.digest(p_ip,'sha256'),'hex');
   keyuser := 'user:'||lower(trim(coalesce(p_data->>'username','')));
   perform pg_advisory_xact_lock(hashtext('eego:'||keyip)); perform pg_advisory_xact_lock(hashtext('eego:'||keyuser));
   insert into eego.throttle(key,reset_at) values(keyip,now()+interval '15 minutes'),(keyuser,now()+interval '15 minutes') on conflict(key) do update set failures=case when eego.throttle.reset_at<now() then 0 else eego.throttle.failures end,reset_at=case when eego.throttle.reset_at<now() then now()+interval '15 minutes' else eego.throttle.reset_at end;
   if exists(select 1 from eego.throttle where (key=keyuser and failures>=5) or (key=keyip and failures>=20)) then return '{"status":429,"error":"Too many login attempts. Please wait 15 minutes."}'; end if;
   select * into u from eego.users where lower(username)=lower(trim(p_data->>'username'));
   if u.id is null or extensions.crypt(coalesce(p_data->>'password',''),u.password_hash) is distinct from u.password_hash then
     if u.id is null then perform extensions.crypt('invalid',extensions.gen_salt('bf',12)); end if;
     update eego.throttle set failures=failures+1 where key in(keyip,keyuser);
     return '{"status":401,"error":"Incorrect username or password."}';
   end if;
   update eego.throttle set failures=0 where key=keyuser;
   delete from eego.sessions where expires_at<now();
   delete from eego.throttle where reset_at<now()-interval '1 day';
   token:=encode(extensions.gen_random_bytes(32),'hex');
   insert into eego.sessions(token_hash,user_id,expires_at) values(encode(extensions.digest(token,'sha256'),'hex'),u.id,now()+interval '30 days');
   return jsonb_build_object('status',200,'token',token,'data',jsonb_build_object('username',u.username));
 end if;
 if p_action like 'worker_%' then
   select * into w from eego.workers where enabled and token_hash=encode(extensions.digest(p_token,'sha256'),'hex');
   if w.id is null then return '{"status":401,"error":"Worker authentication required."}'; end if;
   if p_action='worker_heartbeat' then
     update eego.workers set last_seen=now(),ready=coalesce((p_data->>'ready')::boolean,false) where id=w.id;
     return '{"status":200,"data":{"ok":true}}';
   end if;
   if p_action='worker_claim' then
     if not w.ready then return '{"status":409,"error":"Codex login is not ready."}'; end if;
     update eego.workers set last_seen=now() where id=w.id;
     update eego.jobs set state='failed',error='worker_timeout',finished_at=now() where user_id=w.user_id and state='running' and lease_until<now() and tries>=3;
     select * into j from eego.jobs where user_id=w.user_id and (state='queued' or (state='running' and lease_until<now())) and tries<3 order by created_at for update skip locked limit 1;
     if j.id is null then return '{"status":200,"data":{"job":null}}'; end if;
     token:=encode(extensions.gen_random_bytes(24),'hex');
     update eego.jobs set state='running',tries=tries+1,lease_token=token,lease_until=now()+interval '12 minutes',worker_id=w.id where id=j.id;
     select jsonb_agg(jsonb_build_object('id',i.id,'kind',i.kind,'level',i.level,'label',i.label,'meaningJa',i.meaning_ja,'notesJa',i.notes_ja)) into arr from eego.items i where id=any(j.targets);
     select coalesce(jsonb_agg(t.sentence),'[]') into v from (select sentence from eego.questions where item_id=any(j.targets) order by created_at desc limit 50)t;
     return jsonb_build_object('status',200,'data',jsonb_build_object('job',jsonb_build_object('id',j.id,'lease',token,'count',j.count,'topic',j.topic,'targets',arr,'avoidSentences',v)));
   end if;
   select * into j from eego.jobs where id=(p_data->>'jobId')::uuid and worker_id=w.id and state='running' and lease_token=p_data->>'lease' and lease_until>now() for update;
   if j.id is null then return '{"status":409,"error":"Job lease expired or does not belong to this worker."}'; end if;
   if p_action='worker_fail' then
     update eego.jobs set state='failed',finished_at=now(),error=case when p_data->>'reason' in('auth_required','quota','validation','timeout') then p_data->>'reason' else 'worker_error' end where id=j.id;
     return '{"status":200,"data":{"ok":true}}';
   end if;
   if p_action='worker_complete' then
     arr:=p_data->'questions';
     if jsonb_typeof(arr) is distinct from 'array' or jsonb_array_length(arr)<>j.count then return '{"status":422,"error":"Question count does not match the requested batch."}'; end if;
     for v in select value from jsonb_array_elements(arr) loop
       if not coalesce((v->>'itemId')=any(j.targets),false) or length(coalesce(v->>'sentence','')) not between 15 and 600 or (length(v->>'sentence')-length(replace(v->>'sentence','____','')))<>4 or jsonb_typeof(v->'options') is distinct from 'array' then return '{"status":422,"error":"Invalid target, sentence or options."}'; end if;
       if jsonb_array_length(v->'options')<>4 or (select count(distinct lower(trim(value))) from jsonb_array_elements_text(v->'options'))<>4 or not (v->'options' ? (v->>'answer')) or exists(select 1 from jsonb_array_elements_text(v->'options') where length(value) not between 1 and 160) or length(coalesce(v->>'explanationJa','')) not between 5 and 1500 or length(coalesce(v->>'translationJa','')) not between 3 and 1000 or length(coalesce(v->>'hintJa','')) not between 1 and 300 or not (v->>'explanationJa' ~ '[ぁ-んァ-ヶ一-龯]') then return '{"status":422,"error":"Invalid answer or Japanese explanation."}'; end if;
       if exists(select 1 from eego.questions where fingerprint=encode(extensions.digest(lower(regexp_replace(trim(v->>'sentence'),'\s+',' ','g')),'sha256'),'hex')) then return '{"status":422,"error":"Duplicate sentence. Generate a genuinely new context."}'; end if;
     end loop;
     if (select count(distinct lower(regexp_replace(trim(value->>'sentence'),'\s+',' ','g'))) from jsonb_array_elements(arr))<>j.count then return '{"status":422,"error":"Duplicate sentences within batch."}'; end if;
     for v in select value from jsonb_array_elements(arr) loop
       insert into eego.questions(item_id,sentence,options,answer,hint_ja,explanation_ja,translation_ja,source,job_id,fingerprint) values(v->>'itemId',v->>'sentence',v->'options',v->>'answer',v->>'hintJa',v->>'explanationJa',v->>'translationJa','Codex generated; automatically checked, not human reviewed',j.id,encode(extensions.digest(lower(regexp_replace(trim(v->>'sentence'),'\s+',' ','g')),'sha256'),'hex'));
     end loop;
     update eego.jobs set state='completed',finished_at=now(),question_count=j.count where id=j.id;
     return jsonb_build_object('status',200,'data',jsonb_build_object('saved',j.count));
   end if;
   return '{"status":404,"error":"Unknown worker operation."}';
 end if;
 select user_id into uid from eego.sessions where token_hash=encode(extensions.digest(p_token,'sha256'),'hex') and expires_at>now();
 if uid is null then return '{"status":401,"error":"Please log in."}'; end if;
 select * into u from eego.users where id=uid;
 if p_action='connect_worker' then
   if exists(select 1 from eego.jobs where user_id=uid and state='running' and lease_until>now()) then return '{"status":409,"error":"Wait for the current Codex batch to finish before reconnecting."}'; end if;
   token:=encode(extensions.gen_random_bytes(32),'hex');
   insert into eego.workers(id,user_id,token_hash,enabled,ready,last_seen)
   values('user-'||uid::text,uid,encode(extensions.digest(token,'sha256'),'hex'),true,false,null)
   on conflict(id) do update set token_hash=excluded.token_hash,enabled=true,ready=false,last_seen=null;
   return jsonb_build_object('status',200,'data',jsonb_build_object('workerToken',token,'workerId','user-'||uid::text));
 end if;
 if p_action='logout' then delete from eego.sessions where token_hash=encode(extensions.digest(p_token,'sha256'),'hex'); return '{"status":200,"data":{"ok":true}}'; end if;
 if p_action='password' then
   if length(coalesce(p_data->>'newPassword','')) not between 10 and 128 then return '{"status":400,"error":"Choose a new password with 10 to 128 characters."}'; end if;
   if extensions.crypt(coalesce(p_data->>'currentPassword',''),u.password_hash) is distinct from u.password_hash then return '{"status":401,"error":"Current password is incorrect."}'; end if;
   update eego.users set password_hash=extensions.crypt(p_data->>'newPassword',extensions.gen_salt('bf',12)) where id=uid;
   delete from eego.sessions where user_id=uid and token_hash<>encode(extensions.digest(p_token,'sha256'),'hex');
   return '{"status":200,"data":{"ok":true}}';
 end if;
 if p_action in('me','dashboard') then
   select jsonb_build_object('username',u.username,'totalAnswers',(select count(*) from eego.attempts where user_id=uid),'correctAnswers',(select count(*) from eego.attempts where user_id=uid and correct),'todayAnswers',(select count(*) from eego.attempts where user_id=uid and (created_at at time zone 'Asia/Tokyo')::date=(now() at time zone 'Asia/Tokyo')::date),'weakCount',(select count(*) from eego.mastery where user_id=uid and weak),'dueCount',(select count(*) from eego.mastery where user_id=uid and next_due<=now()),'learnedCount',(select count(*) from eego.mastery where user_id=uid and not weak and recovery>=3),'itemCount',(select count(*) from eego.items),'questionCount',(select count(*) from eego.questions where not reported),'workerOnline',exists(select 1 from eego.workers where user_id=uid and enabled and ready and last_seen>now()-interval '90 seconds'),'activeJobCount',(select count(*) from eego.jobs where user_id=uid and state in('queued','running'))) into outjson;
   return jsonb_build_object('status',200,'data',outjson);
 end if;
 if p_action in('catalog','weak') then
   v_kind:=coalesce(p_data->>'kind','all'); lvl:=coalesce(p_data->>'level','all');
   select coalesce(jsonb_agg(to_jsonb(t)),'[]') into arr from (
    select i.*,coalesce(m.weak,false) as weak,coalesce(m.seen,0) as seen,coalesce(m.correct,0) as correct,coalesce(m.recovery,0) as recovery,m.next_due,(select count(*) from eego.questions q where q.item_id=i.id and not q.reported) as question_count
    from eego.items i left join eego.mastery m on m.item_id=i.id and m.user_id=uid
    where (v_kind='all' or i.kind=v_kind) and (lvl='all' or i.level=lvl) and (p_action<>'weak' or m.weak) and (coalesce(p_data->>'search','')='' or strpos(lower(i.label),lower(p_data->>'search'))>0 or strpos(i.meaning_ja,p_data->>'search')>0)
    order by i.level,i.kind,i.label limit 100 offset greatest(0,least(coalesce((p_data->>'offset')::int,0),100000))
   )t;
   return jsonb_build_object('status',200,'data',jsonb_build_object('items',arr));
 end if;
 if p_action='practice' then
   v_kind:=coalesce(p_data->>'kind','all'); lvl:=coalesce(p_data->>'level','all'); mode:=coalesce(p_data->>'mode','all');
   if (select count(*) from eego.practice where user_id=uid and created_at>now()-interval '1 minute')>=12 then return '{"status":429,"error":"Please finish a practice session before starting more."}'; end if;
   select array_agg(t.id) into ids from (
    select qq.id from (select q.id,q.item_id,row_number() over(partition by q.item_id order by (select count(*) from eego.attempts a where a.user_id=uid and a.question_id=q.id),random()) as rn,coalesce(m.weak,false) as weak,coalesce(m.seen,0) as seen,m.next_due
    from eego.questions q join eego.items i on i.id=q.item_id left join eego.mastery m on m.item_id=i.id and m.user_id=uid
    where not q.reported and (q.job_id is null or exists(select 1 from eego.jobs jj where jj.id=q.job_id and jj.user_id=uid)) and (v_kind='all' or i.kind=v_kind) and (lvl='all' or i.level=lvl) and (mode<>'weak' or m.weak) and (mode<>'due' or m.next_due<=now()) and (coalesce(p_data->>'itemId','')='' or i.id=p_data->>'itemId') and (coalesce(p_data->>'jobId','')='' or q.job_id=(p_data->>'jobId')::uuid))qq
    order by case when coalesce(p_data->>'itemId','')='' then qq.rn else 1 end,case when qq.weak then 0 else 1 end,qq.seen,random() limit 10
   )t;
   if ids is null then return '{"status":200,"data":{"questions":[],"message":"No matching questions yet. Try another level or generate a batch."}}'; end if;
   insert into eego.practice(user_id,question_ids) values(uid,ids) returning * into s;
   select jsonb_agg(jsonb_build_object('id',q.id,'kind',i.kind,'level',i.level,'sentence',q.sentence,'options',(select jsonb_agg(value order by random()) from jsonb_array_elements(q.options)),'hintJa',q.hint_ja,'source',q.source) order by array_position(ids,q.id)) into arr from eego.questions q join eego.items i on i.id=q.item_id where q.id=any(ids);
   return jsonb_build_object('status',200,'data',jsonb_build_object('sessionId',s.id,'questions',arr));
 end if;
 if p_action='answer' then
   select * into s from eego.practice where id=(p_data->>'sessionId')::uuid and user_id=uid for update;
   if s.id is null or not coalesce((p_data->>'questionId')::uuid=any(s.question_ids),false) then return '{"status":403,"error":"This question is not part of your session."}'; end if;
   select feedback into fb from eego.attempts where session_id=s.id and question_id=(p_data->>'questionId')::uuid;
   if fb is not null then return jsonb_build_object('status',200,'data',fb); end if;
   select * into q from eego.questions where id=(p_data->>'questionId')::uuid;
   choice:=trim(coalesce(p_data->>'choice','')); unsure:=coalesce((p_data->>'unsure')::boolean,false);
   if not (q.options ? choice) then return '{"status":400,"error":"Please choose one of the four answers."}'; end if;
   ok:=choice=q.answer;
   insert into eego.mastery(user_id,item_id) values(uid,q.item_id) on conflict do nothing;
   select * into m from eego.mastery where user_id=uid and item_id=q.item_id for update;
   step:=m.recovery; isweak:=m.weak; due:=m.next_due;
   if not ok or unsure then step:=0; isweak:=true; due:=now()+interval '10 minutes';
   elsif now()>=m.next_due and m.last_question is distinct from q.id then
     step:=least(4,step+1); if step>=3 then isweak:=false; end if;
     due:=now()+case step when 1 then interval '1 day' when 2 then interval '3 days' when 3 then interval '7 days' else interval '30 days' end;
   end if;
   update eego.mastery set seen=seen+1,correct=correct+case when ok then 1 else 0 end,lapses=lapses+case when not ok or unsure then 1 else 0 end,weak=isweak,recovery=step,next_due=due,last_question=q.id,updated_at=now() where user_id=uid and item_id=q.item_id;
   select jsonb_build_object('correct',ok,'unsure',unsure,'choice',choice,'answer',q.answer,'sentence',q.sentence,'explanationJa',q.explanation_ja,'translationJa',q.translation_ja,'label',i.label,'meaningJa',i.meaning_ja,'notesJa',i.notes_ja,'itemId',i.id,'kind',i.kind,'level',i.level,'weak',isweak,'recovery',step,'nextDue',due,'questionId',q.id) into fb from eego.items i where i.id=q.item_id;
   insert into eego.attempts(user_id,session_id,question_id,item_id,correct,unsure,feedback) values(uid,s.id,q.id,q.item_id,ok,unsure,fb);
   return jsonb_build_object('status',200,'data',fb);
 end if;
 if p_action='history' then
   select coalesce(jsonb_agg(to_jsonb(t)),'[]') into arr from (select id,created_at,feedback from eego.attempts where user_id=uid and (not coalesce((p_data->>'mistakes')::boolean,false) or not correct or unsure) order by created_at desc limit 30 offset greatest(0,least(coalesce((p_data->>'offset')::int,0),100000)))t;
   return jsonb_build_object('status',200,'data',jsonb_build_object('attempts',arr));
 end if;
 if p_action='report' then
   if not exists(select 1 from eego.practice where user_id=uid and (p_data->>'questionId')::uuid=any(question_ids)) then return '{"status":403,"error":"Practice this question first."}'; end if;
   insert into eego.reports(user_id,question_id) values(uid,(p_data->>'questionId')::uuid) on conflict do nothing;
   update eego.questions set reported=true where id=(p_data->>'questionId')::uuid;
   return '{"status":200,"data":{"ok":true}}';
 end if;
 if p_action='generate' then
   perform pg_advisory_xact_lock(hashtext('eego-generation:'||uid::text));
   if (select count(*) from eego.jobs where user_id=uid and created_at>now()-interval '24 hours')>=6 then return '{"status":429,"error":"Daily limit reached: six batches per 24 hours. Saved practice still works."}'; end if;
   if exists(select 1 from eego.jobs where user_id=uid and state in('queued','running')) then return '{"status":409,"error":"A batch is already waiting or running."}'; end if;
   v_kind:=p_data->>'kind'; lvl:=p_data->>'level'; requested:=coalesce((p_data->>'count')::int,10);
   if v_kind not in('vocabulary','grammar') or lvl not in('B1','B2','C1','C2') or requested not in(5,10,20) or coalesce(p_data->>'topic','') not in('daily life','work','travel','mixed') then return '{"status":400,"error":"Choose a valid category, level, topic and batch size."}'; end if;
   select array_agg(t.id) into targets from(select i.id from eego.items i left join eego.mastery m on m.item_id=i.id and m.user_id=uid where i.kind=v_kind and i.level=lvl and (not coalesce((p_data->>'weakOnly')::boolean,false) or m.weak) and (coalesce(p_data->>'itemId','')='' or i.id=p_data->>'itemId') order by coalesce(m.weak,false) desc,coalesce(m.seen,0),random() limit 5)t;
   if targets is null then return '{"status":400,"error":"No matching items. Turn off weak-only or choose another level."}'; end if;
   insert into eego.jobs(user_id,kind,level,count,targets,topic) values(uid,v_kind,lvl,requested,targets,p_data->>'topic') returning id into jobuuid;
   return jsonb_build_object('status',202,'data',jsonb_build_object('jobId',jobuuid,'state','queued','workerOnline',exists(select 1 from eego.workers where user_id=uid and enabled and ready and last_seen>now()-interval '90 seconds')));
 end if;
 if p_action='cancel_job' then
   update eego.jobs set state='failed',error='cancelled',finished_at=now() where id=(p_data->>'jobId')::uuid and user_id=uid and state='queued';
   return '{"status":200,"data":{"ok":true}}';
 end if;
 if p_action='jobs' then
   select coalesce(jsonb_agg(to_jsonb(t)),'[]') into arr from(select id,kind,level,count,state,created_at,finished_at,error,question_count from eego.jobs where user_id=uid order by created_at desc limit 20)t;
   return jsonb_build_object('status',200,'data',jsonb_build_object('jobs',arr));
 end if;
 if p_action='export' then
   return jsonb_build_object('status',200,'data',jsonb_build_object('format','eego-study-export','schemaVersion',1,'exportedAt',now(),'username',u.username,'mastery',coalesce((select jsonb_agg(to_jsonb(t)-'user_id') from eego.mastery t where user_id=uid),'[]'),'history',coalesce((select jsonb_agg(to_jsonb(t)-'user_id') from eego.attempts t where user_id=uid),'[]')));
 end if;
 return '{"status":404,"error":"Unknown operation."}';
end $$;
revoke all on function public.eego_api(text,jsonb,text,text) from public,anon,authenticated;
grant execute on function public.eego_api(text,jsonb,text,text) to service_role;
