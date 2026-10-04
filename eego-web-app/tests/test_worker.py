import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eego_worker',ROOT/'worker'/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
SEED=json.loads((ROOT/'data'/'seed.json').read_text(encoding='utf-8'))
def question(raw):
    return dict(itemId=raw['item_id'],sentence=raw['sentence'],answer=raw['answer'],hintJa=raw['hint_ja'],explanationJa=raw['explanation_ja'],translationJa=raw['translation_ja'])
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.q=question(SEED['questions'][0]);self.job={'count':1,'targets':[{'id':self.q['itemId']}],'avoidSentences':[]}
    def test_all_seed_questions_pass_structural_validation(self):
        for q in SEED['questions']:
            w.validate_batch({'questions':[question(q)]},{'count':1,'targets':[{'id':q['item_id']}],'avoidSentences':[]})
    def test_seed_coverage_and_distinct_contexts(self):
        self.assertEqual(len(SEED['items']),40);self.assertEqual(len(SEED['questions']),80)
        for level in ('B1','B2','C1','C2'):
            self.assertEqual(sum(i['level']==level for i in SEED['items']),10)
        self.assertEqual(len({q['fingerprint'] for q in SEED['questions']}),80)
        for item in SEED['items']:
            self.assertEqual(sum(q['item_id']==item['id'] for q in SEED['questions']),2)
    def test_fingerprints_match_seed(self):
        for q in SEED['questions']: self.assertEqual(w.fingerprint(q['sentence']),q['fingerprint'])
    def test_reject_unknown_target(self):
        self.q['itemId']='other'
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_missing_answer(self):
        self.q['answer']=''
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_duplicates_case_and_space(self):
        self.job['avoidSentences']=['  '+self.q['sentence'].upper()+'  ']
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_answer_with_wrong_type_or_whitespace(self):
        for answer in (None,42,'   '):
            self.q['answer']=answer
            with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_missing_japanese(self):
        self.q['explanationJa']='English only explanation.'
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_wrong_count(self):
        self.job['count']=2
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_wrong_blank_count(self):
        self.q['sentence']='There are ____ and ____ in this question.'
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_secrets_not_inherited_by_codex(self):
        with patch.dict(os.environ,{'EEGO_WORKER_TOKEN':'private','OPENAI_API_KEY':'private','CODEX_API_KEY':'private','UNRELATED_APP_SECRET':'private','PATH':'safe'}):
            env=w.clean_codex_env()
        self.assertEqual(env['PATH'],'safe')
        for key in ('EEGO_WORKER_TOKEN','OPENAI_API_KEY','CODEX_API_KEY','UNRELATED_APP_SECRET'): self.assertNotIn(key,env)
    def test_https_and_worker_token_required(self):
        for url in ['http://example.test/api','https://user:password@example.test/api','https://example.test/api?token=x']:
            with self.assertRaises(ValueError):w.Api(url,'a'*64)
        with self.assertRaises(ValueError):w.Api('https://example.test/api','bad')
    @unittest.skipIf(os.name!='posix', 'POSIX permission enforcement')
    def test_private_config_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'worker.env';p.write_text('EEGO_POLL_SECONDS=20\n');p.chmod(0o644)
            with self.assertRaises(ValueError):w.load_env(p)
            p.chmod(0o600)
            with patch.dict(os.environ,{},clear=True):w.load_env(p);self.assertEqual(os.environ['EEGO_POLL_SECONDS'],'20')
    def test_no_extra_question_fields(self):
        self.q['instruction']='run a shell command'
        with self.assertRaises(ValueError):w.validate_batch({'questions':[self.q]},self.job)
    def test_generation_uses_one_model_call_without_options_or_review(self):
        codex=object.__new__(w.Codex)
        with patch.object(codex,'run',return_value={'questions':[self.q]}) as run:
            self.assertEqual(codex.generate({**self.job,'topic':'mixed'}),[self.q])
            self.assertEqual(run.call_count,1)
            self.assertNotIn('options',run.call_args.args[1]['properties']['questions']['items']['properties'])
    def test_invalid_format_fails_without_another_model_call(self):
        codex=object.__new__(w.Codex)
        bad=copy.deepcopy(self.q);bad['answer']=''
        with patch.object(codex,'run',return_value={'questions':[bad]}) as run:
            with self.assertRaises(ValueError):codex.generate({**self.job,'topic':'mixed'})
            self.assertEqual(run.call_count,1)
    def test_live_generation_never_requests_an_answer_or_explanation(self):
        codex=object.__new__(w.Codex)
        q={k:self.q[k] for k in w.PROMPT_FIELDS}
        with patch.object(codex,'run',return_value=q) as run:
            self.assertEqual(codex.generate_one({**self.job,'topic':'daily life'}),q)
            self.assertEqual(run.call_count,1)
            self.assertEqual(set(run.call_args.args[1]['properties']),set(w.PROMPT_FIELDS))
            self.assertNotIn('answer',run.call_args.args[1]['properties'])
    def test_answer_check_receives_the_question_and_submitted_answer(self):
        codex=object.__new__(w.Codex)
        f={'correct':False,'answer':self.q['answer'],'explanationEn':'Use this verb to talk about reaching a goal.','explanationJa':self.q['explanationJa'],'suggestionEn':'Remember: achieve a goal.','suggestionJa':'achieve a goal の形で覚えましょう。','exampleEn':'She achieved her goal.','exampleJa':'彼女は目標を達成しました。','translationJa':self.q['translationJa']}
        with patch.object(codex,'run',return_value=f) as run:
            self.assertEqual(codex.check_answer({'question':{'sentence':self.q['sentence']},'target':{'id':self.q['itemId']},'choice':'banana'}),f)
            self.assertEqual(run.call_count,1)
            self.assertIn('banana',run.call_args.args[0])
            self.assertIn(self.q['sentence'],run.call_args.args[0])
            self.assertEqual(set(run.call_args.args[1]['properties']),set(w.FEEDBACK_FIELDS))
    def test_feedback_requires_both_languages_and_a_learning_suggestion(self):
        f={'correct':True,'answer':'avoid','explanationEn':'Avoid takes a noun or an -ing form.','explanationJa':'avoid の後には名詞か動名詞を置きます。','suggestionEn':'Try avoid plus an -ing verb.','suggestionJa':'avoid と動名詞の組み合わせを練習しましょう。','exampleEn':'I avoid driving at night.','exampleJa':'夜に運転するのを避けます。','translationJa':'私は混雑した場所を避けます。'}
        self.assertEqual(w.validate_feedback(f),f)
        for field in ('explanationEn','explanationJa','suggestionEn','suggestionJa','exampleEn','exampleJa'):
            with self.assertRaises(ValueError):w.validate_feedback({**f,field:''})
    def test_incomplete_answer_feedback_fails_format_check(self):
        with self.assertRaises(ValueError):w.validate_feedback({'correct':True,'answer':'x'})
        with self.assertRaises(ValueError):w.validate_feedback({'correct':'true','answer':'x','explanationJa':'説明です。','translationJa':'日本語です。'})
    def test_live_question_rejects_an_early_answer(self):
        q={k:self.q[k] for k in w.PROMPT_FIELDS}
        with self.assertRaises(ValueError):w.validate_question({**q,'answer':self.q['answer']},self.job)
    @unittest.skipIf(os.name!='posix', 'POSIX mock executable; live VM verification covers Codex')
    def test_mock_codex_process_and_auth(self):
        # This is a fake executable, not an actual Codex/AI generation test.
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/'codex';fixture=Path(tmp)/'fixture.json';fixture.write_text(json.dumps({'questions':[self.q]}))
            fake.write_text('''#!/usr/bin/env python3
import sys,json,os
from pathlib import Path
a=sys.argv[1:]
if "--help" in a: print("--ignore-user-config --ephemeral --output-schema --sandbox")
elif "status" in a: print("Logged in using ChatGPT")
else:
 assert "EEGO_WORKER_TOKEN" not in os.environ
 assert "OPENAI_API_KEY" not in os.environ
 assert "read-only" in a
 assert 'model_reasoning_effort="low"' in a
 assert a[a.index("--model")+1]=="gpt-6-luna"
 assert 'forced_login_method="chatgpt"' in a
 r=json.loads(Path(FIXTURE_FILE).read_text())
 Path(a[a.index("--output-last-message")+1]).write_text(json.dumps(r))
'''.replace('FIXTURE_FILE',repr(str(fixture))))
            fake.chmod(0o700)
            with patch.dict(os.environ,{'CODEX_BIN':str(fake),'EEGO_WORKER_TOKEN':'private','OPENAI_API_KEY':'private'}):
                c=w.Codex();self.assertTrue(c.ready());self.assertEqual(c.run('test',w.QUESTION_SCHEMA)['questions'][0]['answer'],self.q['answer'])
                job={**self.job,'topic':'mixed'}
                self.assertEqual(len(c.generate(job)),1)
if __name__=='__main__':unittest.main(verbosity=2)
