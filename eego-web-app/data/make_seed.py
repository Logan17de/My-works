"""Original starter lessons. CEFR bands are editorial practice estimates, not certification."""
import json, hashlib, re
from pathlib import Path
items=[]; questions=[]
def add(id,kind,level,label,meaning,notes,rows):
    examples=[]
    for sentence,answer,wrong,ja in rows:
        assert sentence.count('____')==1
        options=[answer,*wrong]
        assert len(options)==len(set(options))==4
        full=sentence.replace('____',answer)
        examples.append({'en':full,'ja':ja})
        questions.append(dict(item_id=id,sentence=sentence,options=options,answer=answer,hint_ja=meaning,explanation_ja=notes,translation_ja=ja,source='Original Eego starter',fingerprint=hashlib.sha256(re.sub(r'\s+',' ',sentence.strip()).lower().encode()).hexdigest()))
    items.append(dict(id=id,kind=kind,level=level,label=label,meaning_ja=meaning,notes_ja=notes,source='Original Eego starter; CEFR band is an editorial estimate',examples=examples))
def v(id,level,label,meaning,notes,rows): add('v.'+id,'vocabulary',level,label,meaning,notes,rows)
def g(id,level,label,meaning,notes,rows): add('g.'+id,'grammar',level,label,meaning,notes,rows)
v('achieve','B1','achieve','達成する','achieve a goal は「目標を達成する」。努力の結果として成果や成功を得るときに使います。',[
('With regular practice, you can ____ your goal of running five kilometres.','achieve',['borrow','spill','postpone'],'定期的に練習すれば、5キロ走るという目標を達成できます。'),
('After months of work, the team finally ____ the results it had hoped for.','achieved',['lent','spilled','mislaid'],'数か月の努力の末、チームはついに期待していた成果を達成しました。')])
v('afford','B1','afford','経済的・時間的な余裕がある','can afford + 名詞 / to不定詞。「費用などを負担できる」という意味です。否定形では「〜する余裕がない」。',[
('The rent is too high; I cannot ____ to live there.','afford',['borrow','invite','repair'],'家賃が高すぎて、そこに住む余裕がありません。'),
('We saved for a year so that we could ____ a new washing machine.','afford',['spill','interrupt','lend'],'新しい洗濯機を買えるように、1年間貯金しました。')])
v('avoid','B1','avoid','避ける','avoid + 名詞 / 動名詞。「avoid to do」ではなく「avoid doing」を使います。',[
('Leave early to ____ getting stuck in traffic.','avoid',['borrow','lend','afford'],'渋滞に巻き込まれないよう、早めに出発してください。'),
('She ____ looking at her phone during the interview.','avoided',['repaired','borrowed','lent'],'彼女は面接中にスマートフォンを見るのを避けました。')])
v('borrow','B1','borrow','借りる','borrow something from someone は「人から物を借りる」。lend は逆方向の「貸す」です。',[
('May I ____ your umbrella? I will return it tomorrow.','borrow',['lend','achieve','avoid'],'傘を借りてもいいですか。明日返します。'),
('I ____ three books from the library yesterday.','borrowed',['lent','achieved','avoided'],'昨日、図書館で本を3冊借りました。')])
v('improve','B1','improve','改善する・上達する','improve は他動詞でも自動詞でも使えます。「improve your English」は英語力を伸ばすことです。',[
('Reading every day can help you ____ your vocabulary.','improve',['spill','lend','interrupt'],'毎日の読書は語彙力を伸ばす助けになります。'),
('Her pronunciation has ____ considerably since last year.','improved',['borrowed','lent','spilled'],'彼女の発音は去年から大きく上達しました。')])
v('prefer','B1','prefer','より好む','prefer A to B は「BよりAを好む」。prefer の比較対象には通常 than ではなく to を使います。',[
('I ____ tea to coffee because coffee keeps me awake.','prefer',['repair','lend','achieve'],'コーヒーを飲むと眠れなくなるので、私はコーヒーよりお茶が好きです。'),
('She ____ walking to taking a crowded bus.','prefers',['borrows','spills','lends'],'彼女は混んだバスに乗るより歩くほうが好きです。')])
v('acknowledge','B2','acknowledge','認める・受け取ったことを知らせる','acknowledge は事実や責任を認めるほか、連絡や書類の受領を知らせる意味でも使います。',[
('The manager finally ____ that the mistake was his responsibility.','acknowledged',['borrowed','mislaid','spilled'],'上司はついに、そのミスが自分の責任だと認めました。'),
('Please ____ receipt of this message by replying to it.','acknowledge',['borrow','achieve','spill'],'このメッセージを受信したことを、返信でお知らせください。')])
v('approach','B2','approach · method','取り組み方・方法','an approach to + 名詞 / 動名詞で「〜への取り組み方」。この項目では名詞の意味を練習します。',[
('We need a different ____ to solving this problem.','approach',['receipt','ceiling','departure'],'この問題を解決するには、別の取り組み方が必要です。'),
('Her practical ____ helped the team turn an idea into a working product.','approach',['receipt','ceiling','umbrella'],'彼女の実践的な取り組み方が、アイデアを動く製品にする助けとなりました。')])
v('consequence','B2','consequence','結果・影響','as a consequence は「その結果」。consequence は行動や出来事から生じる結果を表します。',[
('A possible ____ of missing the deadline is losing the contract.','consequence',['ingredient','receipt','ceiling'],'締め切りに遅れると、契約を失う可能性があります。'),
('He ignored the warnings and had to face the ____ of his decision.','consequences',['ingredients','umbrellas','receipts'],'彼は警告を無視したため、自分の決断の結果に向き合わなければなりませんでした。')])
v('maintain','B2','maintain','維持する・手入れする','maintain は状態や水準を保つこと。設備を正常な状態に保つという意味もあります。',[
('It takes regular exercise to ____ a good level of fitness.','maintain',['borrow','spill','postpone'],'良い体力水準を維持するには、定期的な運動が必要です。'),
('The machines must be properly ____ to prevent breakdowns.','maintained',['borrowed','spilled','forgotten'],'故障を防ぐには、機械を適切に整備しなければなりません。')])
v('reluctant','B2','reluctant','気が進まない・消極的な','be reluctant to do は「〜する気が進まない」。完全な拒否ではなく、ためらいを表します。',[
('She was ____ to lend him more money because he had not repaid the last loan.','reluctant',['identical','rectangular','fluent'],'彼が前の借金を返していなかったため、彼女はさらにお金を貸すことに気が進みませんでした。'),
('He agreed to speak, but his ____ tone suggested he would rather remain silent.','reluctant',['triangular','identical','wooden'],'彼は話すことに同意しましたが、その消極的な口調からは黙っていたい気持ちがうかがえました。')])
v('resolve','B2','resolve · solve','解決する','resolve a problem / dispute は「問題・争いを解決する」。解決に至ることに重点があります。',[
('The two sides met to ____ their disagreement.','resolve',['borrow','spill','lend'],'双方は意見の相違を解決するために会いました。'),
('The technical issue was ____ before the presentation began.','resolved',['borrowed','spilled','lent'],'技術的な問題は発表が始まる前に解決されました。')])
v('allocate','C1','allocate','割り当てる','allocate resources to a task は「作業に資源を割り当てる」。時間・予算・人員にも使います。',[
('The committee will ____ additional funds to the research project.','allocate',['contradict','resemble','hesitate'],'委員会は研究プロジェクトに追加資金を割り当てます。'),
('We have ____ two hours for questions after the workshop.','allocated',['contradicted','resembled','hesitated'],'ワークショップの後、質問のために2時間を割り当てました。')])
v('concede.admit','C1','concede · admit','しぶしぶ認める','concede that ... は、議論や証拠を受けて「〜だと認める」。譲歩するニュアンスがあります。',[
('After seeing the evidence, he had to ____ that his estimate was wrong.','concede',['allocate','resemble','hesitate'],'証拠を見た後、彼は自分の見積もりが間違っていたと認めざるを得ませんでした。'),
('She ____ that the proposal had some merit, despite her initial objections.','conceded',['allocated','resembled','hesitated'],'彼女は当初反対していましたが、その提案にも一定の価値があると認めました。')])
v('concede.goals','C1','concede · allow a goal','失点する','スポーツで concede a goal / point は「ゴール・得点を相手に許す」。議論で認める意味とは区別します。',[
('The team ____ two goals in the final ten minutes.','conceded',['allocated','resembled','hesitated'],'そのチームは最後の10分間に2失点しました。'),
('A strong defence helped the visitors avoid ____ a single goal.','conceding',['allocating','resembling','hesitating'],'堅い守備のおかげで、アウェーチームは1点も失わずに済みました。')])
v('compelling','C1','compelling','説得力のある・強く引きつける','compelling evidence / argument は「説得力のある証拠・主張」。物語が魅力的で目を離せない場合にも使います。',[
('The scientist presented ____ evidence that challenged the old theory.','compelling',['rectangular','edible','identical'],'科学者は従来の理論に疑問を投げかける、説得力のある証拠を示しました。'),
('The novel was so ____ that I read it in one sitting.','compelling',['edible','triangular','identical'],'その小説は非常に引きつけられる内容で、私は一気に読みました。')])
v('feasible','C1','feasible','実行可能な','feasible は条件や資源を考えて実行できること。「望ましい」ことと「実行可能」なことは別です。',[
('The plan is attractive, but it is not financially ____.','feasible',['edible','audible','identical'],'その計画は魅力的ですが、資金面では実行できません。'),
('We need to decide whether finishing the work by Friday is ____.','feasible',['edible','rectangular','fluent'],'金曜日までに作業を終えることが実行可能か判断する必要があります。')])
v('mitigate','C1','mitigate','悪影響・深刻さを和らげる','mitigate a risk / impact は「リスク・影響を軽減する」。完全に取り除くという意味とは限りません。',[
('The new flood barriers are designed to ____ the impact of heavy rain.','mitigate',['resemble','borrow','hesitate'],'新しい防水壁は大雨の影響を軽減するために設計されています。'),
('Early action could have ____ some of the damage.','mitigated',['resembled','borrowed','hesitated'],'早く対応していれば、被害の一部を軽減できたかもしれません。')])
v('equivocal','C2','equivocal','曖昧な・どちらとも取れる','equivocal は、意味や立場がはっきりせず複数の解釈ができること。明確な賛否を避ける返答などに使います。',[
('His ____ reply left us unsure whether he supported the proposal.','equivocal',['unequivocal','unanimous','conclusive'],'彼の曖昧な返答では、その提案を支持しているのか分かりませんでした。'),
('The experimental results were ____, so no firm conclusion could be drawn.','equivocal',['conclusive','definitive','unambiguous'],'実験結果ははっきりせず、確かな結論を出せませんでした。')])
v('meticulous','C2','meticulous','細部まで非常に注意深い','meticulous attention to detail は「細部への細心の注意」。手順や作業の丁寧さを表します。',[
('Her ____ records included the date, time and conditions of every measurement.','meticulous',['careless','haphazard','perfunctory'],'彼女の綿密な記録には、すべての測定の日時と条件が含まれていました。'),
('The restoration required ____ attention to the smallest details.','meticulous',['careless','haphazard','fleeting'],'修復には、最も細かな部分まで細心の注意を払う必要がありました。')])
v('ostensibly','C2','ostensibly','表向きは・一見したところは','ostensibly は、公に示された目的や見かけが真の事情と異なる可能性を示します。',[
('He travelled there ____ for a conference, though his real purpose was to visit family.','ostensibly',['unanimously','meticulously','inadvertently'],'彼は表向きは会議のためにそこへ行きましたが、本当の目的は家族に会うことでした。'),
('The rule was ____ introduced for safety, but critics suspected another motive.','ostensibly',['unanimously','alphabetically','physically'],'その規則は表向きは安全のために導入されましたが、批判する人々は別の動機を疑いました。')])
v('perfunctory','C2','perfunctory','おざなりの・形式的な','perfunctory は、関心や努力をほとんど払わず形だけ行うこと。丁寧で徹底した行為の反対です。',[
('He gave the document a ____ glance instead of reading it carefully.','perfunctory',['meticulous','thorough','exhaustive'],'彼は書類を注意深く読む代わりに、おざなりに目を通しただけでした。'),
('Her apology sounded ____, as though she only wanted to end the conversation.','perfunctory',['heartfelt','sincere','contrite'],'彼女の謝罪は、会話を終わらせたいだけのようで、形式的に聞こえました。')])
v('tenuous','C2','tenuous','根拠・つながりが弱い','a tenuous link / claim は「弱いつながり・根拠の乏しい主張」。物理的な細さを表す場合もあります。',[
('The connection between the two events is too ____ to support that conclusion.','tenuous',['robust','definitive','conclusive'],'その2つの出来事のつながりは弱すぎて、その結論を裏づけられません。'),
('Their argument rests on a ____ assumption that has never been tested.','tenuous',['well-established','conclusive','watertight'],'彼らの議論は、一度も検証されていない、根拠の弱い仮定に基づいています。')])
v('ubiquitous','C2','ubiquitous','至る所にある','ubiquitous は、ある物がいたる所で見られること。単に「人気がある」という意味ではありません。',[
('Smartphones have become so ____ that it is unusual to meet someone without one.','ubiquitous',['scarce','obsolete','inaccessible'],'スマートフォンは至る所に普及し、持っていない人に会うほうが珍しくなりました。'),
('The brand\'s ____ logo appeared on buses, billboards and shop windows.','ubiquitous',['invisible','obsolete','illegible'],'そのブランドのロゴは、バスや看板、店の窓など至る所に見られました。')])
g('present-perfect','B1','Present perfect','過去から現在につながる経験・状態','have / has + 過去分詞。since は起点を表し、現在まで続く状態に現在完了を使います。',[
('She ____ in this town since 2020 and still lives here.','has lived',['lives','is living','will live'],'彼女は2020年から現在までこの町に住んでいます。'),
('I ____ that film three times so far.','have seen',['see','am seeing','will see'],'私はこれまでにその映画を3回見ました。')])
g('second-conditional','B1','Second conditional','現在・未来の仮定','If + 過去形, would + 動詞の原形。現実とは異なる状況や実現可能性が低いことを想像します。',[
('If I ____ more free time, I would learn to play the piano.','had',['have','will have','am having'],'もっと自由な時間があれば、ピアノを習うのですが。'),
('If she lived closer, we ____ each other more often.','would see',['will saw','have see','would saw'],'彼女がもっと近くに住んでいれば、もっと頻繁に会えるのですが。')])
g('used-to','B1','Used to','以前はよく〜した・以前は〜だった','used to + 動詞の原形で、現在はそうではない過去の習慣や状態を表します。be used to doing「〜に慣れている」と区別します。',[
('I ____ walk to school every day, but now I take the train.','used to',['am used to','use for','was use to'],'以前は毎日学校まで歩いていましたが、今は電車に乗っています。'),
('There ____ be a cinema here before it became a supermarket.','used to',['is used to','use for','was use'],'スーパーになる前、ここには映画館がありました。')])
g('relative-clauses','B1','Relative clauses','名詞を説明する関係詞節','who は人、which は物を説明します。主格の関係代名詞は、後ろの節の主語の役割を果たします。',[
('The woman ____ lives next door is a doctor.','who',['which','where','when'],'隣に住んでいる女性は医師です。'),
('This is the machine ____ broke down yesterday.','which',['who','where','when'],'これが昨日故障した機械です。')])
g('past-perfect','B2','Past perfect','過去のある時点より前の出来事','had + 過去分詞。2つの過去の出来事のうち、先に起きたことを明確にします。',[
('By the time we arrived at the station, the train ____.','had left',['has left','will leave','leaves'],'私たちが駅に着いたときには、電車はすでに出発していました。'),
('She could not open the door because she ____ her key at work.','had left',['has left','will leave','leaves'],'職場に鍵を置いてきたため、彼女はドアを開けられませんでした。')])
g('third-conditional','B2','Third conditional','実際には起きなかった過去の仮定','If + had + 過去分詞, would have + 過去分詞。過去の事実とは違う状況と、その結果を想像します。',[
('If I ____ about the meeting, I would have attended.','had known',['know','have known','will know'],'その会議について知っていれば、出席したのですが。'),
('If they had left earlier, they ____ the bus.','would have caught',['will catch','would caught','have catch'],'もっと早く出発していれば、彼らはバスに間に合ったのですが。')])
g('passive','B2','Present perfect passive','現在完了の受動態','has / have been + 過去分詞。行為を受ける物や人を主語にして、完了したことなどを表します。',[
('The reports ____ by the editor and are ready to publish.','have been checked',['has been checked','have been checking','have checking'],'報告書は編集者による確認が済み、公開できる状態です。'),
('The bridge ____ repaired, so it is open again.','has been',['have been','has being','have being'],'橋の修理が完了したため、再び通行できます。')])
g('contrast','B2','Although / despite','対比・譲歩の接続','although の後ろは主語と動詞を含む節、despite の後ろは名詞や動名詞です。despite of とは言いません。',[
('____ it was raining heavily, the match continued.','Although',['Despite','In spite of','Because of'],'激しい雨が降っていたにもかかわらず、試合は続きました。'),
('____ the heavy rain, the match continued.','Despite',['Although','Even though','Whereas'],'大雨にもかかわらず、試合は続きました。')])
g('negative-inversion','C1','Negative inversion','否定表現を文頭に置く倒置','Never / Rarely などを文頭に置くと、助動詞 + 主語の語順になります。強調した書き言葉などで使います。',[
('Never ____ such a beautiful night sky.','have I seen',['I have seen','I seen have','have seen I'],'これほど美しい夜空を見たことはありません。'),
('Rarely ____ a proposal that satisfies everyone.','do we find',['we do find','we find do','find do we'],'全員を満足させる提案が見つかることはめったにありません。')])
g('mixed-conditional','C1','Mixed conditional','過去の仮定と現在の結果','If + had + 過去分詞, would + 原形は、過去の事実が違っていたら現在はどうなっているかを表します。',[
('If I had studied medicine, I ____ a doctor now.','would be',['would have been yesterday','will been','had be'],'医学を学んでいたら、今ごろ医師になっていたでしょう。'),
('If she ____ the job offer last year, she would be living in London now.','had accepted',['accepts','will accept','has accept'],'去年その内定を受けていたら、彼女は今ロンドンに住んでいるでしょう。')])
g('cleft','C1','It-cleft sentences','特定の部分を強調する分裂文','It is / was + 強調部分 + that ...。文の一部を取り出して焦点を当てます。',[
('It was the noise ____ woke me up, not the light.','that',['what','where','whose'],'私を起こしたのは光ではなく、音でした。'),
('It was yesterday ____ she finally received the letter.','that',['what','whose','where'],'彼女がようやくその手紙を受け取ったのは昨日でした。')])
g('modal-perfect','C1','Modal perfect','過去への推量・後悔','must have + 過去分詞は過去についての強い推量。should have + 過去分詞は、するべきだったのにしなかったことを表せます。',[
('The lights are off and nobody answers. They must ____ home already.','have gone',['had go','has gone','having go'],'明かりは消え、返事もありません。彼らはもう帰宅したに違いありません。'),
('You should ____ me before changing the schedule.','have told',['had tell','has told','having tell'],'予定を変える前に、私に知らせるべきでした。')])
g('inverted-conditional','C2','Inverted conditionals','if を省略した仮定の倒置','Had + 主語 + 過去分詞は If + 主語 + had + 過去分詞に相当します。形式的な文体で使われます。',[
('____ I known about the delay, I would have stayed at home.','Had',['Have','Did','Would'],'遅延について知っていれば、家にいたのですが。'),
('Had the team prepared more carefully, it ____ the error.','would have avoided',['will avoid','would avoided','has avoiding'],'チームがもっと慎重に準備していれば、そのミスを避けられたでしょう。')])
g('concessive-as','C2','Concessive as','形容詞を前に出す譲歩','形容詞 + as + 主語 + 動詞で「〜だけれども」。Tired as she was は Although she was tired に相当します。',[
('Tired ____ she was, she stayed to finish the work.','as',['despite','because of','in spite of'],'彼女は疲れていましたが、仕事を終えるために残りました。'),
('Difficult ____ the task seemed, they decided to attempt it.','as',['despite','because of','in spite of'],'その課題は難しそうでしたが、彼らは挑戦することにしました。')])
g('not-until','C2','Not until inversion','「〜して初めて」の倒置','Not until ... を文頭に置く場合、倒置するのは主節です。until 節の語順は通常のままです。',[
('Not until I reached home ____ that my wallet was missing.','did I realise',['I did realise','I realised','realised did I'],'家に着いて初めて、財布がないことに気づきました。'),
('Not until the report was published ____ the full extent of the problem.','did we understand',['we understood','we did understand','understood did we'],'報告書が公開されて初めて、私たちは問題の全容を理解しました。')])
g('perfect-participle','C2','Perfect participle clauses','主節より前の動作を表す分詞構文','Having + 過去分詞は、主節より先に完了した動作を示します。分詞構文と主節の意味上の主語は通常同じです。',[
('____ the report, she sent it to her manager.','Having finished',['Having finish','Have finishing','Having been finish'],'報告書を書き終えてから、彼女はそれを上司に送りました。'),
('____ all the options, we chose the least expensive one.','Having considered',['Having consider','Have considering','Having been consider'],'すべての選択肢を検討したうえで、最も安いものを選びました。')])
root=Path(__file__).parent
(root/'seed.json').write_text(json.dumps({'items':items,'questions':questions},ensure_ascii=False,indent=2))
payload=json.dumps({'items':items,'questions':questions},ensure_ascii=False)
sql="""-- Original Eego starter. Apply only after the deployment/import block is resolved.
-- Idempotent, additive seed: existing lessons and learning history are not overwritten.
DO $seed$
DECLARE d jsonb := $eego$"""+payload+"""$eego$::jsonb;
BEGIN
  INSERT INTO eego.items(id,kind,level,label,meaning_ja,notes_ja,source,examples)
  SELECT x.id,x.kind,x.level,x.label,x.meaning_ja,x.notes_ja,x.source,x.examples
  FROM jsonb_to_recordset(d->'items') x(id text,kind text,level text,label text,meaning_ja text,notes_ja text,source text,examples jsonb)
  ON CONFLICT(id) DO NOTHING;
  INSERT INTO eego.questions(item_id,sentence,options,answer,hint_ja,explanation_ja,translation_ja,source,fingerprint)
  SELECT x.item_id,x.sentence,x.options,x.answer,x.hint_ja,x.explanation_ja,x.translation_ja,x.source,x.fingerprint
  FROM jsonb_to_recordset(d->'questions') x(item_id text,sentence text,options jsonb,answer text,hint_ja text,explanation_ja text,translation_ja text,source text,fingerprint text)
  ON CONFLICT(fingerprint) DO NOTHING;
END $seed$;
"""
(root/'seed.sql').write_text(sql)
print(len(items),'items;',len(questions),'questions;',len(sql),'SQL chars')
