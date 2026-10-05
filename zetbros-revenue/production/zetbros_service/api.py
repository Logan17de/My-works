"""Bearer-only JSON API. There is no model-facing decision tool or browser UI."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import AuthenticationError
from .config import load_settings
from .models import DecisionInput, Identifier, ProposalInput, ReconciliationInput
from .service import Service
from .store import StoreError


MAX_BODY = 32768


def create_app(service: Service, *, run_worker: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = 4
        stop = asyncio.Event()
        task = asyncio.create_task(service.worker_loop(stop)) if run_worker else None
        yield
        stop.set()
        if task:
            await task  # In-flight submissions finish or time out before graceful stop.

    app = FastAPI(title="Zetbros single-deployment support reply",version="0.1.0",
                  docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        if any("token" in key.lower() or "authorization" in key.lower() for key in request.query_params):
            return JSONResponse({"error":"tokens_in_urls_forbidden"},status_code=400)
        if request.url.path.startswith("/v1/"):
            # No cookies, browser form endpoints, CSRF flow or CORS authorization.
            if request.headers.get("origin") is not None:
                return JSONResponse({"error":"browser_origin_not_supported"},status_code=403)
            if len(request.headers.getlist("authorization")) != 1:
                return JSONResponse({"error":"authentication_required"},status_code=401)
            try:
                request.state.principal = service.verifier.verify(request.headers.get("authorization"))
                service.store.rate_limit(request.state.principal)
            except AuthenticationError:
                return JSONResponse({"error":"authentication_required"},status_code=401,headers={"WWW-Authenticate":"Bearer"})
            except StoreError as exc:
                return JSONResponse({"error":exc.code},status_code=exc.status)
            if request.method in ("POST","PUT","PATCH"):
                if request.headers.get("content-type","").split(";")[0].strip() != "application/json":
                    return JSONResponse({"error":"json_required"},status_code=415)
                try:
                    content_length = int(request.headers.get("content-length","0"))
                except ValueError:
                    return JSONResponse({"error":"invalid_content_length"},status_code=400)
                if content_length < 0 or content_length > MAX_BODY:
                    return JSONResponse({"error":"payload_too_large"},status_code=413)
                async def read_body():
                    chunks, size = [], 0
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_BODY:
                            raise StoreError("payload_too_large",413)
                        chunks.append(chunk)
                    return b"".join(chunks)
                def unique_object(pairs):
                    value = {}
                    for key,item in pairs:
                        if key in value:
                            raise ValueError("duplicate JSON key")
                        value[key] = item
                    return value
                try:
                    body = await asyncio.wait_for(read_body(),timeout=5)
                    json.loads(body,object_pairs_hook=unique_object,parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON number")))
                    request._body = body
                except asyncio.TimeoutError:
                    return JSONResponse({"error":"body_timeout"},status_code=408)
                except StoreError as exc:
                    return JSONResponse({"error":exc.code},status_code=exc.status)
                except (ValueError,UnicodeDecodeError):
                    return JSONResponse({"error":"invalid_json"},status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(StoreError)
    async def storage_error(request, exc):
        return JSONResponse({"error":exc.code,"request_id":request.state.request_id},status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Default framework details can echo bodies. Return a fixed redacted error.
        return JSONResponse({"error":"invalid_schema","request_id":request.state.request_id},status_code=422)

    @app.get("/healthz")
    def health():
        return {"status":"up"}

    @app.get("/readyz")
    def ready():
        result = service.readiness()
        return JSONResponse(result,status_code=200 if result["control_ready"] else 503)

    @app.get("/v1/source/{source_id}")
    def read(source_id: Identifier, request: Request):
        return service.read_source(source_id)

    @app.post("/v1/proposals",status_code=201)
    def propose(body: ProposalInput, request: Request):
        return service.propose(body,request.state.principal,request.state.request_id)

    @app.get("/v1/proposals/{proposal_id}")
    def proposal(proposal_id: uuid.UUID, request: Request):
        return service.view(str(proposal_id),request.state.principal)

    @app.post("/v1/reviews/{proposal_id}/decision")
    def decide(proposal_id: uuid.UUID, body: DecisionInput, request: Request):
        actor = request.state.principal
        if actor.role != "reviewer":
            raise StoreError("reviewer_required",403)
        # The reviewer must GET the exact preview and return its complete digest.
        return service.store.decide(str(proposal_id),body.digest,body.decision,actor,request.state.request_id)

    @app.post("/v1/reviews/{proposal_id}/reconciliation")
    def reconcile(proposal_id: uuid.UUID, body: ReconciliationInput, request: Request):
        return service.store.reconcile(str(proposal_id),body.digest,body.observed_submission,
                                       body.evidence_reference,request.state.principal,request.state.request_id)

    @app.get("/v1/reviews/{proposal_id}/reconciliations")
    def reconciliations(proposal_id: uuid.UUID, request: Request):
        if request.state.principal.role != "reviewer":
            raise StoreError("reviewer_required",403)
        service.store.get(str(proposal_id))
        return {"observations":service.store.reconciliations(str(proposal_id))}

    @app.get("/v1/audit")
    def audit(request: Request, after: Annotated[int,Query(ge=0)] = 0):
        if request.state.principal.role != "reviewer":
            raise StoreError("reviewer_required",403)
        return {"events":service.store.audit(after)}

    if service.settings.review_origin is not None:
        from .review_ui import install_review_ui
        install_review_ui(app, service, service.settings.review_origin)
    return app


def configured_app() -> FastAPI:
    # No developer auth, token creation, credentials or live transport startup.
    from .pilot_runtime import build_service
    return create_app(build_service(load_settings()))
