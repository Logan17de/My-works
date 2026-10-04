#!/usr/bin/env python3
"""Connect this machine's saved Codex login to your Eego account. No API key."""
import argparse
import getpass
import http.cookiejar
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from worker import Codex

API = 'https://jxvabaqswqembehxligi.supabase.co/functions/v1/eego-api/eego'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username',default='Mayuna')
    parser.add_argument('--env',type=Path,default=Path.home()/'.config/eego/worker.env')
    parser.add_argument('--start',action='store_true',help='Keep processing requested batches after connection')
    parser.add_argument('--once',action='store_true',help='Process at most one queued batch after connection')
    args=parser.parse_args()
    if args.start and args.once: parser.error('Choose --start or --once.')
    cli=Codex()
    if not cli.ready(): raise ValueError('Run codex login on this machine using your ChatGPT account, then retry.')
    target=args.env.expanduser().resolve()
    repo=Path(__file__).resolve().parents[1]
    if target.is_relative_to(repo): raise ValueError('Store the worker configuration outside the repository.')
    if target.exists(): raise ValueError('Configuration already exists. Use worker.py --env with that file, or explicitly select a new --env path.')
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Create exclusively before rotating the server token; never overwrite another configuration.
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.close(fd)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*_): return None
    jar=http.cookiejar.CookieJar()
    opener=urllib.request.build_opener(NoRedirect,urllib.request.HTTPCookieProcessor(jar))
    def call(action,data=None):
        request=urllib.request.Request(API,data=json.dumps({'action':action,'data':data or {}}).encode(),headers={'Content-Type':'application/json','X-Eego':'1'},method='POST')
        with opener.open(request,timeout=30) as response: return json.load(response)
    saved=False
    try:
        call('login',{'username':args.username,'password':getpass.getpass('Eego password: ')})
        result=call('connect_worker')
        token=result.get('workerToken','')
        if len(token)!=64 or any(c not in '0123456789abcdef' for c in token): raise ValueError('Invalid worker registration response.')
        lines=[f'EEGO_API_URL={API}',f'EEGO_WORKER_TOKEN={token}','EEGO_POLL_SECONDS=20']
        for key in ('CODEX_HOME','CODEX_BIN','CODEX_MODEL'):
            value=os.environ.get(key,'')
            if value and '\n' not in value and '\r' not in value: lines.append(f'{key}={value}')
        target.write_text('\n'.join(lines)+'\n')
        saved=True
    finally:
        try: call('logout')
        except Exception: pass
        if not saved: target.unlink(missing_ok=True)
    print(f'Connected to Eego. Private configuration saved to {target}.')
    command=[sys.executable,str(Path(__file__).with_name('worker.py')),'--env',str(target)]
    if args.start or args.once:
        if args.once: command.append('--once')
        raise SystemExit(subprocess.call(command))
    print('Start the worker with:')
    print(subprocess.list2cmdline(command))

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: raise SystemExit(130)
    except Exception as error:
        # Do not expose HTTP bodies, credentials, or model output.
        print(str(error) if isinstance(error,ValueError) else 'Connection failed. Check your Eego login and network, then retry.',file=sys.stderr)
        raise SystemExit(1)
