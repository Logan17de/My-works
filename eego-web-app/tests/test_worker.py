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
SEED=json.loads((ROOT/'data'/'seed.json').read_text())
def question(raw):
    return dict(itemId=raw['item_id'],sentence=raw['sentence'],options=raw['options'],answer=raw['answer'],hintJa=raw['hint_ja'],explanationJa=raw['explanation_ja'],translationJa=raw['translation_ja'])
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
        self.q['answer']='not an option'
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_duplicates_case_and_space(self):
        self.job['avoidSentences']=['  '+self.q['sentence'].upper()+'  ']
        with self.assertRaises(ValueError): w.validate_batch({'questions':[self.q]},self.job)
    def test_reject_duplicate_options(self):
        self.q['options'][1]=' '+self.q['options'][0].upper()+' '
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
        with patch.dict(os.environ,{'EEGO_WORKER_TOKEN':'private','OPENAI_API_KEY':'private','CODEX_API_KEY':'private','SUPABASE_SERVICE_ROLE_KEY':'private','PATH':'safe'}):
            env=w.clean_codex_env()
        self.assertEqual(env['PATH'],'safe')
        for key in ('EEGO_WORKER_TOKEN','OPENAI_API_KEY','CODEX_API_KEY','SUPABASE_SERVICE_ROLE_KEY'): self.assertNotIn(key,env)
    def test_https_and_worker_token_required(self):
        for url in ['http://example.test/api','https://user:password@example.test/api','https://example.test/api?token=x']:
            with self.assertRaises(ValueError):w.Api(url,'a'*64)
        with self.assertRaises(ValueError):w.Api('https://example.test/api','bad')
    def test_private_config_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'worker.env';p.write_text('EEGO_POLL_SECONDS=20\n');p.chmod(0o644)
            with self.assertRaises(ValueError):w.load_env(p)
            p.chmod(0o600)
            with patch.dict(os.environ,{},clear=True):w.load_env(p);self.assertEqual(os.environ['EEGO_POLL_SECONDS'],'20')
    def test_no_extra_question_fields(self):
        self.q['instruction']='run a shell command'
        with self.assertRaises(ValueError):w.validate_batch({'questions':[self.q]},self.job)
    def test_mock_codex_process_and_auth(self):
        # This is a fake executable, not an actual Codex/AI generation test.
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/'codex';fixture=Path(tmp)/'fixture.json';fixture.write_text(json.dumps({'questions':[self.q]}))
            fake.write_text('#!/usr/bin/env python3\nimport sys,json,os\nfrom pathlib import Path\na=sys.argv[1:]\nif "--help" in a: print("--ignore-user-config --ephemeral --output-schema --sandbox")\nelif "status" in a: print("Logged in using ChatGPT")\nelse:\n assert "EEGO_WORKER_TOKEN" not in os.environ\n assert "OPENAI_API_KEY" not in os.environ\n assert "read-only" in a\n assert "forced_login_method=\\"chatgpt\\"" in a\n s=json.loads(Path(a[a.index("--output-schema")+1]).read_text())\n r={"valid":True,"issues":[]} if "valid" in s["properties"] else json.loads(Path('+repr(str(fixture))+').read_text())\n Path(a[a.index("--output-last-message")+1]).write_text(json.dumps(r))\n')
            fake.chmod(0o700)
            with patch.dict(os.environ,{'CODEX_BIN':str(fake),'EEGO_WORKER_TOKEN':'private','OPENAI_API_KEY':'private'}):
                c=w.Codex();self.assertTrue(c.ready());self.assertEqual(c.run('test',w.QUESTION_SCHEMA)['questions'][0]['answer'],self.q['answer'])
                job={**self.job,'topic':'mixed'}
                self.assertEqual(len(c.generate(job)),1)
if __name__=='__main__':unittest.main(verbosity=2)
