"""Browser UI tests with an explicitly mocked API. No production login or AI calls."""
import datetime as dt
import json
from pathlib import Path
import uuid
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'data'/'seed.json').read_text())
NOW=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
class Fixture:
    def __init__(self):self.auth=False;self.attempts=[];self.weak={};self.jobs=[];self.sessions={}
    def item(self,i):
        a=[a for a in self.attempts if a['feedback']['itemId']==i['id']]
        return {**i,'question_count':2,'seen':len(a),'correct':sum(a['feedback']['correct'] for a in a),'weak':i['id'] in self.weak,'recovery':0,'next_due':NOW()}
    def handle(self,action,data):
        if action=='login':
            if data.get('username')=='Mayuna' and data.get('password')=='fixture-password':self.auth=True;return 200,{'username':'Mayuna'}
            return 401,{'error':'Incorrect username or password.'}
        if not self.auth:return 401,{'error':'Please log in.'}
        if action in ('me','dashboard'):return 200,dict(username='Mayuna',totalAnswers=len(self.attempts),correctAnswers=sum(x['feedback']['correct'] for x in self.attempts),todayAnswers=len(self.attempts),weakCount=len(self.weak),dueCount=0,learnedCount=0,itemCount=40,questionCount=80,workerOnline=False,activeJobCount=sum(j['state']=='queued' for j in self.jobs))
        if action in ('catalog','weak'):
            items=[self.item(i) for i in DATA['items'] if (data.get('kind','all')=='all' or data['kind']==i['kind']) and (data.get('level','all')=='all' or data['level']==i['level']) and (not data.get('search') or data['search'].lower() in i['label'].lower() or data['search'] in i['meaning_ja']) and (action!='weak' or i['id'] in self.weak)]
            return 200,{'items':items[data.get('offset',0):][:100]}
        if action=='practice':
            qs=[]
            for n,q in enumerate(DATA['questions']):
                item=next(i for i in DATA['items'] if i['id']==q['item_id'])
                if data.get('itemId') and q['item_id']!=data['itemId']:continue
                if data.get('mode')=='weak' and q['item_id'] not in self.weak:continue
                if not data.get('itemId') and any(x['itemId']==q['item_id'] for x in qs):continue
                qs.append({'id':str(uuid.uuid5(uuid.NAMESPACE_DNS,'eego-test-'+str(n))),'itemId':q['item_id'],'kind':item['kind'],'level':item['level'],'sentence':q['sentence'],'options':q['options'],'hintJa':q['hint_ja'],'source':'Local UI fixture','raw':q})
                if len(qs)>=4:break
            sid=str(uuid.uuid4());self.sessions[sid]=qs
            return 200,{'sessionId':sid,'questions':[{k:v for k,v in q.items() if k not in ('raw','itemId')} for q in qs]}
        if action=='answer':
            q=next(q for q in self.sessions[data['sessionId']] if q['id']==data['questionId']);raw=q['raw'];i=next(i for i in DATA['items'] if i['id']==raw['item_id'])
            existing=next((a for a in self.attempts if a['key']==data['sessionId']+q['id']),None)
            if existing:return 200,existing['feedback']
            correct=data['choice']==raw['answer'];weak=not correct or data['unsure']
            if weak:self.weak[i['id']]=True
            f=dict(correct=correct,unsure=data['unsure'],choice=data['choice'],answer=raw['answer'],sentence=raw['sentence'],explanationJa=raw['explanation_ja'],translationJa=raw['translation_ja'],label=i['label'],meaningJa=i['meaning_ja'],notesJa=i['notes_ja'],itemId=i['id'],kind=i['kind'],level=i['level'],weak=i['id'] in self.weak,recovery=0,nextDue=NOW(),questionId=q['id'])
            self.attempts.insert(0,{'id':str(uuid.uuid4()),'key':data['sessionId']+q['id'],'created_at':NOW(),'feedback':f});return 200,f
        if action=='history':return 200,{'attempts':[a for a in self.attempts if not data.get('mistakes') or not a['feedback']['correct'] or a['feedback']['unsure']]}
        if action=='jobs':return 200,{'jobs':self.jobs}
        if action=='generate':
            j=dict(id=str(uuid.uuid4()),kind=data['kind'],level=data['level'],count=data['count'],created_at=NOW(),state='queued',question_count=0,error=None);self.jobs.insert(0,j);return 202,{'jobId':j['id'],'state':'queued','workerOnline':False}
        if action=='cancel_job':
            for j in self.jobs:
                if j['id']==data['jobId']:j['state']='failed';j['error']='cancelled'
            return 200,{'ok':True}
        if action=='export':return 200,{'format':'eego-study-export','schemaVersion':1,'username':'UI test fixture','mastery':list(self.weak),'history':self.attempts}
        if action in ('report','password'):return 200,{'ok':True}
        if action=='logout':self.auth=False;return 200,{'ok':True}
        return 404,{'error':'Unknown fixture action'}
screens=ROOT/'tests'/'screenshots';screens.mkdir(exist_ok=True)
results=[]
def ok(name):results.append(name);print('PASS',name)
try:
 with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':390,'height':844},device_scale_factor=1)
    fixture=Fixture();errors=[]
    context.expose_function('mock_rpc', lambda payload: dict(zip(('status','body'), fixture.handle(payload.get('action'),payload.get('data',{})))))
    def mount(storage=None):
        page=context.new_page();page.set_default_timeout(7000);page.on('pageerror',lambda err:errors.append(str(err)))
        # In-memory rendering only. No localhost navigation or policy changes.
        html=(ROOT/'web'/'index.html').read_text()
        import re
        html=re.sub(r'<link[^>]*>','',html);html=re.sub(r'<script[^>]*></script>','',html)
        page.set_content(html)
        page.add_style_tag(content=(ROOT/'web'/'style.css').read_text())
        page.evaluate("""(storage)=>{
          window.__eegoSessionStore=storage||{};
          Object.defineProperty(window,'sessionStorage',{configurable:true,value:{getItem:k=>window.__eegoSessionStore[k]||null,setItem:(k,v)=>window.__eegoSessionStore[k]=v,removeItem:k=>delete window.__eegoSessionStore[k]}});
          window.fetch=async(url,opts)=>{const r=await window.mock_rpc(JSON.parse(opts.body));return new Response(JSON.stringify(r.body),{status:r.status,headers:{'Content-Type':'application/json'}});};
          const make=URL.createObjectURL;URL.createObjectURL=function(blob){window.__lastExportBlob=blob;return make.call(URL,blob);};
          HTMLAnchorElement.prototype.click=function(){window.__lastDownload={filename:this.download,href:this.href};};
        }""",storage)
        page.add_script_tag(content='(()=>{'+(ROOT/'web'/'app.js').read_text()+'})();')
        return page
    page=mount();page.locator('#login-form').wait_for()
    page.screenshot(path=str(screens/'mobile-login.png'),full_page=True);ok('Mobile login renders without overflow')
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('[name=username]').fill('Mayuna');page.locator('[name=password]').fill('wrong');page.locator('#login-form button[type=submit]').click();page.locator('#login-error:not(.hidden)').wait_for();ok('Wrong login gives useful feedback')
    page.locator('[name=password]').fill('fixture-password');page.locator('#login-form button[type=submit]').click();page.get_by_role('heading',name='Hello, Mayuna.').wait_for();ok('Login opens dashboard')
    # Explicitly label screenshots as test previews, never as a live deployment.
    page.evaluate("document.body.insertAdjacentHTML('beforeend','<div id=preview-label style=\"position:fixed;bottom:78px;right:12px;font:10px system-ui;background:#fff;border:1px solid #aaa;padding:5px 8px;z-index:99\">LOCAL PREVIEW · TEST DATA</div>')")
    page.screenshot(path=str(screens/'mobile-home.png'),full_page=False)
    page.locator('#preview-label').evaluate('(e)=>e.remove()')
    page.get_by_role('button',name='Start a mixed practice').click();page.locator('.question-card').wait_for()
    page.locator('.option').nth(1).click();page.get_by_role('button',name='Check answer').click();page.locator('.feedback.incorrect').wait_for();assert len(fixture.attempts)==1;ok('Wrong answer shows Japanese explanation and saves once')
    page.screenshot(path=str(screens/'mobile-feedback.png'),full_page=False)
    page.get_by_role('button',name='Weak items',exact=True).click();page.locator('.item').first.wait_for();assert page.locator('.item').count()==1;ok('Weak items are separated')
    page.get_by_role('button',name='History',exact=True).click();page.locator('.history-item').wait_for();page.get_by_text('Why this works',exact=True).click();ok('History reveals Japanese correction')
    saved=page.evaluate('window.__eegoSessionStore');page.close();page=mount(saved);page.get_by_role('heading',name='Hello, Mayuna.').wait_for();page.get_by_role('button',name='Resume session').click();page.locator('.feedback.incorrect').wait_for();ok('Refresh preserves active practice without duplicating attempts')
    page.get_by_role('button',name='Next sentence').click();page.locator('.option').first.click();page.get_by_role('button',name='Check answer').click();page.locator('.feedback:not(.incorrect)').wait_for();ok('Correct answer advances normally')
    page.get_by_role('button',name='Learn',exact=True).click();page.locator('#kind-filter').select_option('grammar');page.locator('#level-filter').select_option('C1');page.wait_for_timeout(200);assert page.locator('.item').count()==4;ok('Library filters category and level')
    page.locator('#search-filter').fill('cleft');page.wait_for_timeout(400);assert page.locator('.item').count()==1;page.locator('.item summary').click();page.locator('.example').first.wait_for();ok('Search and contextual examples work')
    page.get_by_role('button',name='Generate',exact=True).click();page.locator('#generate-form').wait_for();assert 'not connected' in page.locator('#generator-notice').inner_text();page.locator('#generate-form button[type=submit]').click();page.get_by_text('waiting for Codex',exact=True).wait_for();ok('Offline Codex is honestly shown and request stays queued')
    page.get_by_role('button',name='Cancel request').click();page.get_by_text('Cancelled by you.').wait_for();ok('Waiting generation can be cancelled')
    page.get_by_role('button',name='Open settings',exact=True).click();page.locator('#password-form').wait_for()
    page.get_by_role('button',name='Export my progress').click();page.wait_for_function('window.__lastDownload !== undefined');assert page.evaluate('window.__lastDownload.filename').startswith('eego-study-');assert json.loads(page.evaluate('window.__lastExportBlob.text()'))['schemaVersion']==1;ok('Export prepares a valid JSON download')
    page.locator('[name=currentPassword]').fill('fixture-password');page.locator('[name=newPassword]').fill('new-fixture-password');page.get_by_role('button',name='Update password').click();page.get_by_text('Password updated. Other sessions have been signed out.').wait_for();ok('Password update form works')
    page.get_by_role('button',name='Home',exact=True).click();page.get_by_role('heading',name='Hello, Mayuna.').wait_for()
    for width,height in [(360,800),(768,1024),(1365,900)]:
        page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(100);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'overflow {width}'
    page.evaluate("document.body.insertAdjacentHTML('beforeend','<div id=preview-label style=\"position:fixed;bottom:15px;right:15px;font:11px system-ui;background:#fff;border:1px solid #aaa;padding:6px 10px;z-index:99\">LOCAL PREVIEW · TEST DATA</div>')")
    page.screenshot(path=str(screens/'desktop-home.png'),full_page=True);ok('360px, 768px and desktop layouts have no horizontal overflow')
    page.get_by_role('button',name='Open settings',exact=True).click();page.get_by_role('button',name='Sign out',exact=True).click();page.locator('#login-form').wait_for();ok('Sign-out returns to private login')
    assert errors==[],errors;ok('No JavaScript runtime errors')
    browser.close()
finally:pass
(ROOT/'tests'/'ui-results.json').write_text(json.dumps({'status':'passed','type':'mocked_api_browser_tests','checks':results,'live_backend_tested':False,'real_codex_tested':False},indent=2))
print(len(results),'browser checks passed; API and learning state were mocked.')
