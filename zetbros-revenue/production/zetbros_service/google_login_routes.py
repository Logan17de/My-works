"""Dedicated GIS ID-token exchange, never an ordinary bearer/API credential path."""
import asyncio
import json
from http.cookies import SimpleCookie
from fastapi import Request
from fastapi.responses import JSONResponse
from .auth import AuthenticationError
from .google_reviewer_auth import CHALLENGE_COOKIE,SESSION_COOKIE


def unique_cookie(request,name):
    raw=request.headers.getlist('cookie')
    if len(raw)>1:raise AuthenticationError()
    if not raw:return None
    pairs=raw[0].split(';')
    matching=[part for part in pairs if part.strip().partition('=')[0]==name]
    if len(matching)!=1:raise AuthenticationError()
    parsed=SimpleCookie();parsed.load(matching[0].strip())
    if set(parsed)!={name}:raise AuthenticationError()
    return parsed[name].value


def install_google_login_routes(app,service,auth):
    origin=auth.profile.origin
    def secure(request,*,post=False):
        if str(request.base_url).rstrip('/')!=origin:raise AuthenticationError()
        if request.headers.get('authorization') is not None:raise AuthenticationError()
        if request.headers.get('origin') is not None and request.headers.get('origin')!=origin:raise AuthenticationError()
        if post and request.headers.get('origin')!=origin:raise AuthenticationError()
        if len(request.headers.getlist('origin'))>1:raise AuthenticationError()
        if request.headers.get('sec-fetch-site') not in (None,'none','same-origin'):raise AuthenticationError()
        if request.query_params:raise AuthenticationError()
    def cookie(response,name,value,seconds):
        response.set_cookie(name,value,max_age=seconds,httponly=True,secure=True,samesite='lax',path='/review')
    def failed():return JSONResponse({'error':'reviewer_authentication_failed'},status_code=401)
    @app.get('/review/auth/bootstrap')
    def bootstrap(request:Request):
        try:
            secure(request)
            previous=None
            if request.cookies.get(CHALLENGE_COOKIE) is not None:
                previous=unique_cookie(request,CHALLENGE_COOKIE)
            if request.client is None:raise AuthenticationError()
            # Transport peer only, never a raw attacker-controlled forwarding header.
            token,nonce,csrf=auth.challenge(previous,request.client.host)
            response=JSONResponse({'mode':'google_oidc','client_id':auth.profile.client_id,'nonce':nonce,'csrf':csrf})
            cookie(response,CHALLENGE_COOKIE,token,auth.profile.challenge_seconds)
            return response
        except AuthenticationError:return failed()
    @app.post('/review/auth/google')
    async def google(request:Request):
        try:
            secure(request,post=True)
            if request.headers.get('content-type','').split(';')[0].strip()!='application/json':raise AuthenticationError()
            if len(request.headers.getlist('x-login-csrf'))!=1:raise AuthenticationError()
            chunks=[];size=0
            async def body():
                nonlocal size
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>12288:raise AuthenticationError()
                    chunks.append(chunk)
                return b''.join(chunks)
            raw=await asyncio.wait_for(body(),timeout=5)
            def unique(pairs):
                value={}
                for key,item in pairs:
                    if key in value:raise AuthenticationError()
                    value[key]=item
                return value
            value=json.loads(raw,object_pairs_hook=unique)
            if type(value)is not dict or set(value)!={'credential'}:raise AuthenticationError()
            challenge=unique_cookie(request,CHALLENGE_COOKIE)
            token,csrf=await asyncio.to_thread(auth.login,challenge,request.headers['x-login-csrf'],value['credential'])
            response=JSONResponse({'authenticated':True,'csrf':csrf})
            cookie(response,SESSION_COOKIE,token,auth.profile.session_seconds)
            response.delete_cookie(CHALLENGE_COOKIE,path='/review',httponly=True,secure=True,samesite='lax')
            return response
        except Exception:return failed()
    @app.get('/review/auth/session')
    def session(request:Request):
        try:
            secure(request)
            session=auth.session(unique_cookie(request,SESSION_COOKIE))
            return {'authenticated':True,'csrf':session.csrf}
        except AuthenticationError:return failed()
    @app.post('/review/auth/logout')
    def logout(request:Request):
        try:
            secure(request,post=True)
            if len(request.headers.getlist('x-review-csrf'))!=1:raise AuthenticationError()
            auth.logout(unique_cookie(request,SESSION_COOKIE),request.headers['x-review-csrf'])
            response=JSONResponse({'authenticated':False})
            response.delete_cookie(SESSION_COOKIE,path='/review',httponly=True,secure=True,samesite='lax')
            return response
        except AuthenticationError:return failed()
