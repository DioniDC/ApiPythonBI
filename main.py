from __future__ import annotations

"""
SG BI – API (refactor)
- Pool de conexões MySQL
- Autenticação por Bearer (env API_TOKEN)
- Padronização de paginação, validação de datas e erros
- Novas rotas úteis (meta/schema, vendas por filial, clientes inativos, produtos rotação lenta)
- /produtos/sem-movimento aceita dias_sem_movimento OU data_ini/data_fim e estoque opcional

Observação: este arquivo único mantém tudo junto para facilitar teste. Em produção,
recomendo modularizar: config.py, db.py, deps.py, utils.py, routers/*.py
"""

import os
import math
import time
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import uvicorn
from mysql.connector import pooling, Error
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

# ======================== CONFIG ========================
DB_HOST = os.getenv("DB_HOST", "192.168.2.101")
DB_PORT = int(os.getenv("DB_PORT", "8000"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "SG515t3m45")
DB_NAME = os.getenv("DB_NAME", "sgbi_teste")
API_TOKEN = os.getenv("API_TOKEN", "token_criado_apenas_teste_sem_ele_nao_funfa_direito")

DEFAULT_FILIAL = int(os.getenv("DEFAULT_FILIAL", "1"))

# ======================== LOGGING ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sgbi.api")

# ======================== DB (POOL) ========================
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "8"))
_connection_pool: Optional[pooling.MySQLConnectionPool] = None

def init_pool() -> None:
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pooling.MySQLConnectionPool(
            pool_name="sgbi_pool",
            pool_size=POOL_SIZE,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            autocommit=True,
        )
        logger.info("MySQL connection pool inicializado (size=%s)", POOL_SIZE)


def get_conn():
    if _connection_pool is None:
        init_pool()
    assert _connection_pool is not None
    try:
        conn = _connection_pool.get_connection()
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter conexão: {e}")
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ======================== UTILS ========================

def rows_to_dicts(cursor) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    out = []
    for row in cursor.fetchall():
        item = {}
        for c, v in zip(cols, row):
            if isinstance(v, (datetime, date)):
                item[c] = v.isoformat()
            elif isinstance(v, (int, float)):
                item[c] = v
            elif v is None:
                item[c] = None
            else:
                try:
                    item[c] = float(v)
                except Exception:
                    item[c] = str(v)
        out.append(item)
    return out


def paginate(total: int, limit: int, offset: int) -> Dict[str, Any]:
    limit = max(1, min(1000, limit))
    offset = max(0, offset)
    pages = math.ceil(total / limit) if limit else 1
    return {"total": total, "limit": limit, "offset": offset, "pages": pages}


# ======================== MODELS ========================
class Page(BaseModel):
    total: int
    limit: int
    offset: int
    pages: int


class PagedResponse(BaseModel):
    page: Page
    data: List[Dict[str, Any]]


class Periodo(BaseModel):
    data_ini: Optional[str] = Field(None, description="YYYY-MM-DD")
    data_fim: Optional[str] = Field(None, description="YYYY-MM-DD")

    @field_validator("data_ini", "data_fim", mode="before")
    @classmethod
    def _valida_data(cls, v: Optional[str]):
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Use o formato YYYY-MM-DD")
        return v

    def ensure_or_days(self, dias_sem_movimento: Optional[int] = None) -> Tuple[str, str]:
        if self.data_ini and self.data_fim:
            return self.data_ini, self.data_fim
        if dias_sem_movimento is None:
            raise HTTPException(status_code=400, detail="Informe data_ini/data_fim ou dias_sem_movimento")
        hoje = date.today()
        ini = (hoje - timedelta(days=dias_sem_movimento)).isoformat()
        fim = hoje.isoformat()
        return ini, fim

class TipoBusca(str, Enum):
    """Tipos de entidades disponíveis para busca por nome."""
    produto = "produto"
    cliente = "cliente"
    departamento = "departamento"
    grupo = "grupo"
    subgrupo = "subgrupo"

# ======================== APP ========================
app = FastAPI(title="SG BI – API", version="2.1.0", description="Rotas de BI (Vendas, Produtos, Clientes, Financeiro, etc.)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
EXEMPT_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def check_token(request: Request, call_next):
    if request.url.path not in EXEMPT_PATHS:
        token = request.headers.get("Authorization")
        if not token or token != f"Bearer {API_TOKEN}":
            return JSONResponse(status_code=401, content={"detail": "Token inválido ou ausente"})
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)
    logger.info("%s %s %s -> %s (%ss)", request.client.host, request.method, request.url.path, response.status_code, duration)
    return response


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception("Erro não tratado")
        return JSONResponse(status_code=500, content={"detail": str(e)})


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi

# ======================== HEALTH ========================
@app.get("/health")
def health():
    return {"status": "ok", "db": DB_HOST, "schema": DB_NAME}

# ======================== META / SCHEMA ========================
@app.get("/meta/tabelas")
def meta_tabelas(conn=Depends(get_conn)):
    sql = (
        "SELECT TABLE_NAME AS nome, TABLE_TYPE AS tipo "
        "FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s "
        "ORDER BY TABLE_NAME"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (DB_NAME,))
        return rows_to_dicts(cur)


@app.get("/meta/schema")
def meta_schema(table: str = Query(..., min_length=1), conn=Depends(get_conn)):
    sql = (
        "SELECT COLUMN_NAME AS coluna, DATA_TYPE AS tipo, IS_NULLABLE AS nulo, COLUMN_KEY AS chave, COLUMN_DEFAULT AS padrao "
        "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s "
        "ORDER BY ORDINAL_POSITION"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (DB_NAME, table))
        data = rows_to_dicts(cur)
        if not data:
            raise HTTPException(status_code=404, detail=f"Tabela/visão '{table}' não encontrada")
        return data

# ======================== HELPERS DE CONSULTA ========================

def run_count_and_data(conn, qcount: str, qdata: str, pcount: Tuple[Any, ...], pdata: Tuple[Any, ...], limit: int, offset: int) -> PagedResponse:
    limit = max(1, min(1000, limit))
    offset = max(0, offset)
    with conn.cursor() as cur:
        cur.execute(qcount, pcount)
        total = (cur.fetchone() or [0])[0] or 0
    with conn.cursor() as cur:
        cur.execute(qdata, pdata)
        data = rows_to_dicts(cur)
    return PagedResponse(page=Page(**paginate(total, limit, offset)), data=data)

# ===== Helper para rotas /vendas/periodo por dimensão =====

class _DimCfg(BaseModel):
    group: Optional[str] = None
    select_dim: Optional[str] = None
    joins: str = ""

def _cfg_dim(dim: str) -> _DimCfg:
    if dim == "dia":
        return _DimCfg(
            group="DATE(l.dataVenda)",
            select_dim="DATE(l.dataVenda) AS chave"
        )

    if dim == "produto":
        # pega descrição do produto
        return _DimCfg(
            group="l.codigo",
            select_dim="l.codigo AS chave, MAX(p.descricao) AS rotulo",
            joins="LEFT JOIN view_produtos p ON p.codigo=l.codigo AND p.filial=l.filial",
        )

    if dim == "cliente":
        # pega nome/razão do cliente
        return _DimCfg(
            group="l.codigoCliente",
            select_dim="l.codigoCliente AS chave, MAX(c.razaoSocial) AS rotulo",
            joins="LEFT JOIN view_17_clientes_geocode c ON c.codigo=l.codigoCliente",
        )

    if dim == "vendedor":
        # se tiver tabela/visão de vendedores, ajuste aqui:
        # return _DimCfg(
        #     group="l.vendedor",
        #     select_dim="l.vendedor AS chave, MAX(v.nome) AS rotulo",
        #     joins="LEFT JOIN view_vendedores v ON v.codigo=l.vendedor",
        # )
        return _DimCfg(
            group="l.vendedor",
            select_dim="l.vendedor AS chave",
            joins="",
        )

    if dim == "departamento":
        # pega descrição do departamento
        return _DimCfg(
            group="l.departamento",
            select_dim="l.departamento AS chave, MAX(d.descricao) AS rotulo",
            joins="LEFT JOIN view_11_departamentos d ON d.codigo=l.departamento",
        )

    if dim == "grupo":
        # pega descrição do grupo (via produtos -> grupos)
        return _DimCfg(
            group="p.grupo",
            select_dim="p.grupo AS chave, MAX(g.descricao) AS rotulo",
            joins=(
                "LEFT JOIN view_produtos p ON p.codigo=l.codigo AND p.filial=l.filial "
                "LEFT JOIN view_11_grupos g ON g.codigo=p.grupo"
            ),
        )

    if dim == "subgrupo":
        # pega descrição do subgrupo (via produtos -> subgrupos)
        return _DimCfg(
            group="p.subgrupo",
            select_dim="p.subgrupo AS chave, MAX(s.descricao) AS rotulo",
            joins=(
                "LEFT JOIN view_produtos p ON p.codigo=l.codigo AND p.filial=l.filial "
                "LEFT JOIN view_11_subgrupos s ON s.codigo=p.subgrupo"
            ),
        )

    raise HTTPException(status_code=400, detail="Dimensão inválida")

def _vendas_periodo_grouped(
    dim: str,
    periodo: Periodo,
    filial: int,
    codigo: Optional[str],
    top: Optional[int],
    limit: int,
    offset: int,
    conn,
) -> PagedResponse:
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")

    cfg = _cfg_dim(dim)
    where = ["l.filial=%s", "l.dataVenda BETWEEN %s AND %s", "l.cupomCancelado=''"]
    params: List[Any] = [filial, periodo.data_ini, periodo.data_fim]

    if dim == "cliente":
        where.append("l.codigoCliente > 0")

    if codigo:
        campo = cfg.select_dim.split(" AS ")[0] if cfg.select_dim else None
        if not campo:
            raise HTTPException(status_code=400, detail="Dimensão sem campo de filtro")
        where.append(f"{campo}=%s")
        params.append(codigo)

    w = " AND ".join(where)

    base_select = (
        f"SELECT {cfg.select_dim}, "
        f"ROUND(SUM(l.valorTotal),2) AS venda, "
        f"SUM(l.quantidadeVendida) AS itens, "
        f"COUNT(DISTINCT CONCAT(l.numeroCaixa,'-',l.numeroCupom,'-',DATE(l.dataVenda))) AS cupons "
        f"FROM logpdv l {cfg.joins} "
        f"WHERE {w} "
        f"GROUP BY {cfg.group} "
        f"ORDER BY venda DESC"
    )

    sql = base_select
    params_tail: Tuple[Any, ...] = tuple(params)

    if codigo:
        sql += " LIMIT 1"
    elif top is not None:
        sql += " LIMIT %s"
        params_tail = tuple([*params, top])
    else:
        sql += " LIMIT %s OFFSET %s"
        params_tail = tuple([*params, limit, offset])

    with conn.cursor() as cur:
        cur.execute(sql, params_tail)
        data = rows_to_dicts(cur)

    if codigo or top is not None:
        total = len(data)
        page = Page(total=total, limit=total or 1, offset=0, pages=1)
        return PagedResponse(page=page, data=data)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM (SELECT {cfg.group} "
            f"FROM logpdv l {cfg.joins} WHERE {w} GROUP BY {cfg.group}) t",
            tuple(params),
        )
        total = (cur.fetchone() or [0])[0] or 0

    return PagedResponse(page=Page(**paginate(total, limit, offset)), data=data)

# ======================== VENDAS ========================
@app.get("/vendas/resumo-diario", response_model=PagedResponse, tags=["Vendas"])
def vendas_resumo_diario(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")
    qcount = (
        "SELECT COUNT(*) FROM ( SELECT DATE(v.data) AS d FROM view_vendia v "
        "WHERE v.filial=%s AND v.data BETWEEN %s AND %s GROUP BY DATE(v.data) ) t"
    )
    qdata = (
        "SELECT DATE(v.data) AS data, COUNT(DISTINCT v.produto) AS produtos_distintos, "
        "SUM(v.quantidade) AS itens, ROUND(SUM(v.valor_total),2) AS venda_bruta, "
        "ROUND(SUM(v.lucro_bruto),2) AS lucro_bruto, ROUND(AVG(v.lucro_sobre_venda),2) AS margem_media "
        "FROM view_vendia v WHERE v.filial=%s AND v.data BETWEEN %s AND %s "
        "GROUP BY DATE(v.data) ORDER BY data LIMIT %s OFFSET %s"
    )
    return run_count_and_data(conn, qcount, qdata, (filial, periodo.data_ini, periodo.data_fim), (filial, periodo.data_ini, periodo.data_fim, limit, offset), limit, offset)


@app.get("/vendas/resumo-filial", tags=["Vendas"])  # usa PARFIL
def vendas_resumo_filial(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")
    sql = (
        "SELECT p.filial, DATE(p.dataVenda) AS data, p.totvrvenda AS venda, p.totprodvda AS itens, "
        "p.nclientes AS cupons, p.sku AS skus, p.totvrcusto AS custo "
        "FROM parfil p WHERE p.filial=%s AND p.dataVenda BETWEEN %s AND %s ORDER BY data"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (filial, periodo.data_ini, periodo.data_fim))
        return rows_to_dicts(cur)


@app.get("/vendas/por-hora", tags=["Vendas"])  # mantém assinatura, mas via Periodo
def vendas_por_hora(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")
    sql = (
        "SELECT HOUR(l.horarioVenda) AS hora, ROUND(SUM(l.valorTotal),2) AS venda, SUM(l.quantidadeVendida) AS itens "
        "FROM logpdv l WHERE l.filial=%s AND l.dataVenda BETWEEN %s AND %s AND l.cupomCancelado='' "
        "GROUP BY HOUR(l.horarioVenda) ORDER BY hora"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (filial, periodo.data_ini, periodo.data_fim))
        return rows_to_dicts(cur)

@app.get("/produtos/sem-venda-e-com-estoque", response_model=PagedResponse, tags=["Produtos"])
def produtos_sem_venda_e_com_estoque(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    estoque: float = Query(0.0, description="Estoque mínimo (usa filtro estrito: estoqueAtual > estoque)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")

    # COUNT
    qcount = (
        "SELECT COUNT(*) "
        "FROM view_produtos p "
        "LEFT JOIN ( "
        "    SELECT DISTINCT codigo, filial "
        "    FROM logpdv "
        "    WHERE dataVenda BETWEEN %s AND %s AND filial=%s AND cupomCancelado='' "
        ") v ON v.codigo=p.codigo AND v.filial=p.filial "
        "WHERE p.filial=%s AND p.estoqueAtual > %s AND v.codigo IS NULL"
    )
    pcount = (periodo.data_ini, periodo.data_fim, filial, filial, estoque)

    # DATA
    qdata = (
        "SELECT p.codigo, p.descricao, p.departamento, p.grupo, p.subgrupo, "
        "       p.precoVenda, p.estoqueAtual, p.custoMedio, p.lucroSobreVenda "
        "FROM view_produtos p "
        "LEFT JOIN ( "
        "    SELECT DISTINCT codigo, filial "
        "    FROM logpdv "
        "    WHERE dataVenda BETWEEN %s AND %s AND filial=%s AND cupomCancelado='' "
        ") v ON v.codigo=p.codigo AND v.filial=p.filial "
        "WHERE p.filial=%s AND p.estoqueAtual > %s AND v.codigo IS NULL "
        "ORDER BY p.descricao "
        "LIMIT %s OFFSET %s"
    )
    pdata = (periodo.data_ini, periodo.data_fim, filial, filial, estoque, limit, offset)

    return run_count_and_data(conn, qcount, qdata, pcount, pdata, limit, offset)

@app.get("/produtos/estoque-abaixo-minimo", response_model=PagedResponse, tags=["Produtos"])
def produtos_estoque_abaixo_minimo(
    filial: int = Query(DEFAULT_FILIAL),
    departamento: Optional[str] = Query(None, description="Código do departamento (opcional)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    where = ["p.filial=%s", "p.estoqueAtual < p.estoqueMinimo"]
    pcount: List[Any] = [filial]
    pdata:  List[Any] = [filial]

    if departamento:
        where.append("p.departamento=%s")
        pcount.append(departamento)
        pdata.append(departamento)

    w = " AND ".join(where)

    qcount = f"SELECT COUNT(*) FROM view_produtos p WHERE {w}"
    qdata = (
        f"SELECT p.codigo, p.descricao, p.departamento, p.grupo, "
        f"       p.estoqueAtual, p.estoqueMinimo, "
        f"       (p.estoqueMinimo - p.estoqueAtual) AS deficit, "
        f"       p.precoVenda, p.custoMedio "
        f"FROM view_produtos p "
        f"WHERE {w} "
        f"ORDER BY deficit DESC, p.descricao "
        f"LIMIT %s OFFSET %s"
    )

    return run_count_and_data(conn, qcount, qdata, tuple(pcount), tuple([*pdata, limit, offset]), limit, offset)

# ROTA 1: /clientes/busca-nome (linha ~673)
@app.get("/clientes/busca-nome", response_model=PagedResponse, tags=["Clientes"])
def clientes_busca_nome(
    nome: str = Query(..., min_length=2, description="Nome ou parte"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    like = f"%{nome.strip()}%"
    qcount = ("SELECT COUNT(*) FROM view_17_clientes_geocode "
              "WHERE razaoSocial LIKE %s OR nomeFantasia LIKE %s")
    qdata = (
        "SELECT codigo, "
        "razaoSocial AS nome, "  # <-- MUDANÇA AQUI
        "nomeFantasia, "
        "CONCAT(SUBSTRING(cpf,1,3),'.***.**-',SUBSTRING(cpf,-2)) AS cpf_mascarado, "
        "telefone, celular, email, cidade, bairro, dataCadastro, dataUltimaCompra "
        "FROM view_17_clientes_geocode "
        "WHERE razaoSocial LIKE %s OR nomeFantasia LIKE %s "
        "ORDER BY razaoSocial LIMIT %s OFFSET %s"
    )
    return run_count_and_data(conn, qcount, qdata,
                              (like, like),
                              (like, like, limit, offset),
                              limit, offset)

# ROTA 2: /busca/cliente (linha ~884) - JÁ ESTÁ CERTO!
@app.get("/busca/cliente", response_model=PagedResponse, tags=["Busca"])
def busca_cliente(
    nome: str = Query(..., min_length=2, description="Razão social (LIKE)"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    like_pattern = f"%{nome.strip().upper()}%"
    qcount = "SELECT COUNT(*) FROM view_17_clientes_geocode WHERE UPPER(razaoSocial) LIKE %s"
    qdata = (
        " SELECT  "
        "   codigo, "
        "   TRIM(razaoSocial) AS nome, "
        "   TRIM(nomeFantasia) AS nomeFantasia, "
        "   TRIM(cpf) AS cpf, "
        "   TRIM(telefone) AS telefone, "
        "   TRIM(email) AS email, "
        "   TRIM(endereco) AS endereco, "
        "   TRIM(bairro)   AS bairro, "
        "   TRIM(cidade)   AS cidade, "
        "   TRIM(uf) AS uf, "
        "   TRIM(cep) AS cep, "
        "   dataCadastro "
        " FROM view_17_clientes_geocode "
        " WHERE UPPER(TRIM(razaoSocial)) LIKE %s "
        " ORDER BY razaoSocial "
        " LIMIT %s OFFSET %s "
    )
    return run_count_and_data(
        conn, qcount, qdata,
        (like_pattern,),
        (like_pattern, limit, offset),
        limit, offset
    )
# ===== NOVAS ROTAS: VENDAS POR DEPARTAMENTO / GRUPO / SUBGRUPO =====
@app.get("/vendas/por-grupo", response_model=PagedResponse, tags=["Vendas"])
def vendas_por_grupo(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Se informado, filtra um grupo específico"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")

    where = ["l.filial=%s", "l.dataVenda BETWEEN %s AND %s", "l.cupomCancelado=''" ]
    params_c: List[Any] = [filial, periodo.data_ini, periodo.data_fim]
    params_d: List[Any] = [filial, periodo.data_ini, periodo.data_fim]

    # JOIN com a tabela de grupos para pegar a descrição
    join = """
        JOIN view_produtos p ON p.codigo=l.codigo AND p.filial=l.filial
        LEFT JOIN view_11_grupos g ON g.codigo=p.grupo
    """

    if codigo:
        where.append("p.grupo=%s")
        params_c.append(codigo)
        params_d.append(codigo)

    w = " AND ".join(where)

    qcount = f"SELECT COUNT(*) FROM (SELECT p.grupo FROM logpdv l {join} WHERE {w} GROUP BY p.grupo) t"
    qdata  = (
        f"SELECT p.grupo AS codigo, "
        f"MAX(g.descricao) AS descricao, "
        f"ROUND(SUM(l.valorTotal),2) AS vendaTotal, "
        f"SUM(l.quantidadeVendida) AS itensVendidos "
        f"FROM logpdv l {join} WHERE {w} "
        f"GROUP BY p.grupo "
        f"ORDER BY vendaTotal DESC LIMIT %s OFFSET %s"
    )
    return run_count_and_data(conn, qcount, qdata, tuple(params_c), tuple([*params_d, limit, offset]), limit, offset)

@app.get("/vendas/por-subgrupo", response_model=PagedResponse, tags=["Vendas"])
def vendas_por_subgrupo(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Se informado, filtra um subgrupo específico"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")

    where = ["l.filial=%s", "l.dataVenda BETWEEN %s AND %s", "l.cupomCancelado=''" ]
    params_c: List[Any] = [filial, periodo.data_ini, periodo.data_fim]
    params_d: List[Any] = [filial, periodo.data_ini, periodo.data_fim]

    # JOIN com a tabela de subgrupos para pegar a descrição
    join = """
        JOIN view_produtos p ON p.codigo=l.codigo AND p.filial=l.filial
        LEFT JOIN view_11_subgrupos s ON s.codigo=p.subgrupo
    """

    if codigo:
        where.append("p.subgrupo=%s")
        params_c.append(codigo)
        params_d.append(codigo)

    w = " AND ".join(where)

    qcount = f"SELECT COUNT(*) FROM (SELECT p.subgrupo FROM logpdv l {join} WHERE {w} GROUP BY p.subgrupo) t"
    qdata  = (
        f"SELECT p.subgrupo AS codigo, "
        f"MAX(s.descricao) AS descricao, "
        f"ROUND(SUM(l.valorTotal),2) AS vendaTotal, "
        f"SUM(l.quantidadeVendida) AS itensVendidos "
        f"FROM logpdv l {join} WHERE {w} "
        f"GROUP BY p.subgrupo "
        f"ORDER BY vendaTotal DESC LIMIT %s OFFSET %s"
    )
    return run_count_and_data(conn, qcount, qdata, tuple(params_c), tuple([*params_d, limit, offset]), limit, offset)

# ===== ROTAS DESMEMBRADAS: VENDAS POR PERÍODO (dimensões específicas) =====

@app.get("/vendas/periodo/total", tags=["Vendas"])
def vendas_periodo_total(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    conn=Depends(get_conn),
):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")

    select_sql = (
        "SELECT ROUND(SUM(l.valorTotal),2) AS venda, SUM(l.quantidadeVendida) AS itens, "
        "COUNT(DISTINCT CONCAT(l.numeroCaixa,'-',l.numeroCupom,'-',DATE(l.dataVenda))) AS cupons "
        "FROM logpdv l WHERE l.filial=%s AND l.dataVenda BETWEEN %s AND %s AND l.cupomCancelado=''"
    )
    with conn.cursor() as cur:
        cur.execute(select_sql, (filial, periodo.data_ini, periodo.data_fim))
        data = rows_to_dicts(cur)
    # retorna objeto único (ou {} se vazio)
    return data[0] if data else {}

@app.get("/vendas/periodo/dia", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_dia(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Filtra por data exata (YYYY-MM-DD)"),
    top: Optional[int] = Query(None, ge=1, le=1000, description="Ranking por venda"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("dia", periodo, filial, codigo, top, limit, offset, conn)

@app.get("/vendas/periodo/produto", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_produto(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Código do produto"),
    top: Optional[int] = Query(None, ge=1, le=1000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("produto", periodo, filial, codigo, top, limit, offset, conn)

@app.get("/vendas/periodo/cliente", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_cliente(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Código do cliente (> 0)"),
    top: Optional[int] = Query(None, ge=1, le=1000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("cliente", periodo, filial, codigo, top, limit, offset, conn)

@app.get("/vendas/periodo/vendedor", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_vendedor(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Código do vendedor"),
    top: Optional[int] = Query(None, ge=1, le=1000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("vendedor", periodo, filial, codigo, top, limit, offset, conn)

@app.get("/vendas/periodo/departamento", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_departamento(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Código do departamento"),
    top: Optional[int] = Query(None, ge=1, le=1000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("departamento", periodo, filial, codigo, top, limit, offset, conn)

@app.get("/vendas/periodo/grupo", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_grupo(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Código do grupo"),
    top: Optional[int] = Query(None, ge=1, le=1000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("grupo", periodo, filial, codigo, top, limit, offset, conn)

@app.get("/vendas/periodo/subgrupo", response_model=PagedResponse, tags=["Vendas"])
def vendas_periodo_subgrupo(
    periodo: Periodo = Depends(),
    filial: int = Query(DEFAULT_FILIAL),
    codigo: Optional[str] = Query(None, description="Código do subgrupo"),
    top: Optional[int] = Query(None, ge=1, le=1000),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    return _vendas_periodo_grouped("subgrupo", periodo, filial, codigo, top, limit, offset, conn)

# ======================== PRODUTOS ========================
@app.get("/produtos/sem-movimento", response_model=PagedResponse, tags=["Produtos"])
def produtos_sem_movimento(
        filial: int = Query(DEFAULT_FILIAL),
        dias: int = Query(30, ge=1, le=365, description="Dias sem venda"),
        departamento: Optional[int] = Query(None, description="Filtrar por departamento (opcional)"),
        limit: int = Query(20, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        conn=Depends(get_conn),
):
    """Produtos com estoque mas sem vendas no período - OTIMIZADO."""

    import time
    start_time = time.time()

    dias = min(dias, 90)
    start_date = (date.today() - timedelta(days=dias)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')

    # ========== ETAPA 1: Buscar códigos que VENDERAM (super rápido) ==========
    query_vendidos = """
        SELECT DISTINCT l.codigo
        FROM logpdv l
        WHERE l.filial = %s
          AND l.dataVenda >= %s
          AND l.dataVenda <= %s
          AND l.cupomCancelado = ''
    """

    with conn.cursor() as cur:
        cur.execute(query_vendidos, (filial, start_date, end_date))
        codigos_vendidos = [row[0] for row in cur.fetchall()]

    logger.info(f"🔍 Encontrados {len(codigos_vendidos)} produtos que venderam em {dias} dias")

    # ========== ETAPA 2: Filtrar produtos usando NOT IN ==========
    where_parts = ["vp.filial = %s", "vp.estoqueAtual > 0"]
    params_base: List[Any] = [filial]

    if departamento is not None:
        where_parts.append("vp.departamento = %s")
        params_base.append(departamento)

    # Se NENHUM produto vendeu, retorna TODOS produtos com estoque
    if codigos_vendidos:
        # Criar placeholders para IN clause
        placeholders = ','.join(['%s'] * len(codigos_vendidos))
        where_parts.append(f"vp.codigo NOT IN ({placeholders})")
        params_base.extend(codigos_vendidos)

    where_sql = " AND ".join(where_parts)

    # COUNT
    qcount = f"SELECT COUNT(*) FROM view_produtos vp WHERE {where_sql}"

    # DATA
    qdata = f"""
        SELECT
            vp.codigo,
            vp.descricao,
            vp.precoVenda,
            vp.estoqueAtual,
            vp.custoMedio,
            vp.lucroSobreVenda,
            vp.departamento
        FROM view_produtos vp
        WHERE {where_sql}
        ORDER BY vp.departamento, vp.descricao
        LIMIT %s OFFSET %s
    """

    pcount = tuple(params_base)
    pdata = tuple([*params_base, limit, offset])

    result = run_count_and_data(conn, qcount, qdata, pcount, pdata, limit, offset)

    elapsed = time.time() - start_time
    logger.info(f"⏱️  Tempo total: {elapsed:.3f}s | Total produtos sem movimento: {result.page.total}")

    return result

@app.get("/produtos/rotacao-lenta", response_model=PagedResponse, tags=["Produtos"])  # exemplo novo
def produtos_rotacao_lenta(
    filial: int = Query(DEFAULT_FILIAL),
    venda_media_max: float = Query(0.05, ge=0.0, description="vendaMediaDiaria ≤ X"),
    estoque_min: float = Query(1.0, ge=0.0, description="estoqueAtual ≥ min"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    qcount = (
        "SELECT COUNT(*) FROM view_produtos WHERE filial=%s AND estoqueAtual>=%s AND vendaMediaDiaria<=%s"
    )
    qdata = (
        "SELECT codigo, descricao, estoqueAtual, vendaMediaDiaria, precoVenda, custoMedio "
        "FROM view_produtos WHERE filial=%s AND estoqueAtual>=%s AND vendaMediaDiaria<=%s "
        "ORDER BY vendaMediaDiaria ASC, descricao LIMIT %s OFFSET %s"
    )
    return run_count_and_data(
        conn,
        qcount,
        qdata,
        (filial, estoque_min, venda_media_max),
        (filial, estoque_min, venda_media_max, limit, offset),
        limit,
        offset,
    )


@app.get("/produtos/preco", tags=["Produtos"])  # igual à sua, mas usa pool/dep
def produtos_preco(codigo: str = Query(...), filial: int = Query(DEFAULT_FILIAL), conn=Depends(get_conn)):
    sql = (
        "SELECT p.codigo, p.descricao, p.precoVenda, p.estoqueAtual, p.custoMedio, p.lucroSobreVenda "
        "FROM view_produtos p WHERE p.filial=%s AND p.codigo=%s LIMIT 1"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (filial, codigo))
        data = rows_to_dicts(cur)
        if not data:
            raise HTTPException(status_code=404, detail="Produto não encontrado")
        return data[0]

# ======================== BUSCAS POR NOME (DESMEMBRADAS) ========================
@app.get("/busca/produto", response_model=PagedResponse, tags=["Busca"])
def busca_produto(
    nome: str = Query(..., min_length=2, description="Nome/descrição do produto (LIKE)"),
    filial: int = Query(DEFAULT_FILIAL),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    like_pattern = f"%{nome.strip().upper()}%"
    qcount = "SELECT COUNT(*) FROM view_produtos WHERE filial=%s AND UPPER(descricao) LIKE %s"
    qdata = (
        "SELECT codigo, descricao, grupo, subgrupo, precoVenda, estoqueAtual, "
        "       custoMedio, lucroSobreVenda "
        "FROM view_produtos "
        "WHERE filial=%s AND UPPER(descricao) LIKE %s "
        "ORDER BY descricao "
        "LIMIT %s OFFSET %s"
    )
    return run_count_and_data(
        conn, qcount, qdata,
        (filial, like_pattern),
        (filial, like_pattern, limit, offset),
        limit, offset
    )

@app.get("/busca/departamento", response_model=PagedResponse, tags=["Busca"])
def busca_departamento(
    nome: str = Query(..., min_length=2, description="Descrição do departamento (LIKE)"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    like_pattern = f"%{nome.strip().upper()}%"
    qcount = (
        "SELECT COUNT(*) FROM view_11_departamentos "
        "WHERE UPPER(descricao) LIKE %s"
    )
    qdata = (
        "SELECT codigo, descricao "
        "FROM view_11_departamentos "
        "WHERE UPPER(descricao) LIKE %s "
        "ORDER BY descricao "
        "LIMIT %s OFFSET %s"
    )
    return run_count_and_data(
        conn, qcount, qdata,
        (like_pattern,),
        (like_pattern, limit, offset),
        limit, offset
    )

@app.get("/busca/grupo", response_model=PagedResponse, tags=["Busca"])
def busca_grupo(
    nome: str = Query(..., min_length=2, description="Descrição do grupo (LIKE)"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    like_pattern = f"%{nome.strip().upper()}%"
    qcount = (
        "SELECT COUNT(*) FROM view_11_grupos "
        "WHERE UPPER(descricao) LIKE %s"
    )
    qdata = (
        "SELECT codigo, descricao "
        "FROM view_11_grupos "
        "WHERE UPPER(descricao) LIKE %s "
        "ORDER BY descricao "
        "LIMIT %s OFFSET %s"
    )
    return run_count_and_data(
        conn, qcount, qdata,
        (like_pattern,),
        (like_pattern, limit, offset),
        limit, offset
    )

@app.get("/busca/subgrupo", response_model=PagedResponse, tags=["Busca"])
def busca_subgrupo(
    nome: str = Query(..., min_length=2, description="Descrição do subgrupo (LIKE)"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    like_pattern = f"%{nome.strip().upper()}%"
    qcount = (
        "SELECT COUNT(*) FROM view_11_subgrupos "
        "WHERE UPPER(descricao) LIKE %s"
    )
    qdata = (
        "SELECT codigo, descricao "
        "FROM view_11_subgrupos "
        "WHERE UPPER(descricao) LIKE %s "
        "ORDER BY descricao "
        "LIMIT %s OFFSET %s"
    )
    return run_count_and_data(
        conn, qcount, qdata,
        (like_pattern,),
        (like_pattern, limit, offset),
        limit, offset
    )

# ======================== CLIENTES ========================
@app.get("/clientes/novos", tags=["Clientes"])  # igual à sua, mas via Periodo
def clientes_novos(periodo: Periodo = Depends(), conn=Depends(get_conn)):
    if not (periodo.data_ini and periodo.data_fim):
        raise HTTPException(status_code=400, detail="Informe data_ini e data_fim")
    sql = (
        "SELECT COUNT(*) AS novos_clientes FROM view_17_clientes_geocode WHERE dataCadastro BETWEEN %s AND %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (periodo.data_ini, periodo.data_fim))
        return rows_to_dicts(cur)


@app.get("/clientes/inativos", response_model=PagedResponse, tags=["Clientes"])  # novo
def clientes_inativos(
    periodo: Periodo = Depends(),
    dias_sem_compra: Optional[int] = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    # Se não vier período, usa dias_sem_compra
    if periodo.data_ini and periodo.data_fim:
        ini, fim = periodo.data_ini, periodo.data_fim
    elif dias_sem_compra is not None:
        ini, fim = Periodo().ensure_or_days(dias_sem_compra)
    else:
        raise HTTPException(status_code=400, detail="Passe data_ini+data_fim OU dias_sem_compra")

    qcount = (
        "SELECT COUNT(*) FROM view_17_clientes_geocode c "
        "WHERE NOT EXISTS (SELECT 1 FROM logpdv l WHERE l.codigoCliente=c.codigo AND l.cupomCancelado='' AND l.dataVenda BETWEEN %s AND %s)"
    )
    qdata = (
        "SELECT c.codigo, c.razaoSocial, c.cidade, c.dataCadastro, c.dataUltimaCompra "
        "FROM view_17_clientes_geocode c "
        "WHERE NOT EXISTS (SELECT 1 FROM logpdv l WHERE l.codigoCliente=c.codigo AND l.cupomCancelado='' AND l.dataVenda BETWEEN %s AND %s) "
        "ORDER BY c.razaoSocial LIMIT %s OFFSET %s"
    )
    return run_count_and_data(conn, qcount, qdata, (ini, fim), (ini, fim, limit, offset), limit, offset)


# ======================== REPLICAÇÃO (igual, com pool) ========================
@app.post("/replicar/dados", tags=["Replicação"])
def replicar_dados(conn=Depends(get_conn)):
    ontem = (date.today() - timedelta(days=1)).isoformat()
    hoje = date.today().isoformat()
    tabelas = [("logpdv", "dataVenda"), ("view_vendia", "data"), ("parfil", "dataVenda"), ("view_analise_gerencial_produtos", "data")]

    try:
        with conn.cursor(dictionary=True) as cur:
            for tabela, campo_data in tabelas:
                cur.execute(f"SELECT * FROM {tabela} WHERE {campo_data}=%s", (ontem,))
                rows = cur.fetchall()
                if not rows:
                    continue
                for row in rows:
                    row[campo_data] = hoje
                    if tabela == "logpdv" and "numeroCupom" in row:
                        row["numeroCupom"] = int(row["numeroCupom"]) + 5000
                    row.pop("id", None)
                    cols = ", ".join(row.keys())
                    placeholders = ", ".join(["%s"] * len(row))
                    insert_sql = f"INSERT INTO {tabela} ({cols}) VALUES ({placeholders})"
                    cur.execute(insert_sql, tuple(row.values()))
            conn.commit()
        return {"status": "ok", "message": f"Dados de {ontem} replicados para {hoje}"}
    except Error as e:
        try:
            conn.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# ======================== RUN ========================
if __name__ == "__main__":
    init_pool()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
