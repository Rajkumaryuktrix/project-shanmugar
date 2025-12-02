# 📄 Tech Choices PRD – Autonomous AI Trader MVP → Scale

## Purpose
Define the MVP technology stack (Option A) and the future-proof alternative OSS stack (Option B), with migration paths that require zero API changes.

## 1. Core Principles

- **OSS-first**: Reduce vendor lock-in
- **Thin adapters**: Wrap DB, LLM, event bus, and auth behind small interfaces to swap without code churn
- **Two-track**: Ship fastest MVP now, keep clear "switch thresholds" for scale
- **Compliance-ready**: Data quality, observability, audit trail from day one

## 2. Stack Summary

| Layer | MVP Choice (Option A) | Scale Choice (Option B) | Adapter / API Contract |
|-------|----------------------|------------------------|------------------------|
| **Frontend** | Streamlit | Next.js (keep Streamlit for ops) | `/api/*` REST/GraphQL |
| **API** | FastAPI | FastAPI (add hot-path Go/Rust svc) | Pydantic models |
| **Realtime workers** | Celery + Redis | Celery + Kafka/Faust | `TaskQueue.enqueue()` |
| **Batch / Orchestration** | Airflow | Dagster / Prefect | `Orchestrator.trigger(job_id)` |
| **App DB + Auth** | Supabase (Postgres) | Postgres + Keycloak | `UserService`, SQLAlchemy models |
| **Market data DB** | ClickHouse | ClickHouse (cluster) | `MarketDataRepo` |
| **Object storage** | S3 (MinIO local) | Ceph / any S3-API | `StorageService.put/get` |
| **Feature store** | Feast (S3/Redis) | Feast (KeyDB/Dynamo) | Feast API |
| **LLM** | LLMService (Groq/OpenAI) | LLMService (vLLM OSS models) | `LLMService.generate()` |
| **Backtesting** | vectorbt + ExecSim | NautilusTrader / Backtrader | `BacktestRunner.run()` |
| **Optimization** | Optuna | Optuna (RDB/distributed) | `Optimizer.optimize()` |
| **Indicators** | pandas-ta + TA-Lib | vectorbt built-ins + numba | `IndicatorLib.calc()` |
| **Data quality** | Great Expectations | Soda Core + Pandera | `DQ.validate(dataset)` |
| **Observability** | Prometheus+Grafana+Loki | SigNoz / Grafana Cloud | OTel traces/metrics/logs |
| **Secrets** | dotenv + GH secrets | Vault / cloud secret mgr | `ConfigService.get_secret()` |
| **CI/CD** | GH Actions + Docker | GH Actions + Argo CD | `deploy()` workflow |
| **Event bus** | Redis Streams | Kafka / Redpanda | `EventBus.publish/subscribe()` |
| **AuthZ (advanced)** | Supabase RLS | Keycloak + OPA/Cedar | JWT claims |

## 3. Switch Thresholds

- **Kafka**: >2M events/day or >48h replay required
- **Postgres+Keycloak**: SSO, data residency, or RLS complexity
- **NautilusTrader**: Need tick-level fills, order book sim, latency modeling
- **vLLM**: LLM spend >15% infra or privacy requirements
- **Dagster/Prefect**: Typed data assets, easier local dev, complex dependencies
- **Vault**: Multi-team, multi-env secrets rotation

## 4. Compliance & Safety From Day One

- **Immutable audit**: ClickHouse + S3 WORM bucket
- **Data quality checks** on all ingests (GX/Soda)
- **Observability** with latency SLO budgets per path
- **LLM output validation** (JSON schema, temp=0)
- **Idempotent task/event processing**

## 5. Migration Map – Option A → Option B

**Principle**: All swaps happen behind service adapters with unchanged public API contracts.

| Component | Option A Impl | Option B Impl | Swap Steps |
|-----------|---------------|---------------|------------|
| **Event bus** | `EventBusRedis` (Redis Streams) | `EventBusKafka` (Kafka/Redpanda) | Implement same publish/subscribe methods; update config/env; redeploy |
| **LLM** | `LLMServiceGroq/OpenAI` | `LLMServiceVLLM` (local models) | Keep `generate(prompt, schema)` signature; swap driver; adjust config for model endpoint |
| **Task queue** | Celery (Redis broker) | Celery (Kafka broker) or Faust | Change broker URL; adjust concurrency in config; no task code changes |
| **Auth** | `UserServiceSupabase` | `UserServiceKeycloak` | Keep same token verify endpoint; update JWT validation logic to Keycloak public key |
| **DB** | SQLAlchemy to Supabase Postgres | SQLAlchemy to self-hosted Postgres/Timescale | Change DB URI; run migration; no model changes |
| **Object storage** | `StorageServiceS3` (MinIO/S3) | `StorageServiceS3` (Ceph/S3) | Change endpoint/credentials; API unchanged |
| **Feature store** | Feast (Redis) | Feast (KeyDB/Dynamo) | Update Feast config; retrain online store; features unchanged |
| **Backtesting** | `BacktestRunnerVectorbt` | `BacktestRunnerNautilus` | Keep method signature; internal engine swap; adjust results adapter |
| **Indicators** | `IndicatorLibPandasTA` | `IndicatorLibVectorbtCustom` | Maintain same `calc()` interface; swap internals |
| **Orchestration** | `AirflowOrchestrator` | `DagsterOrchestrator` | Keep same `trigger(job_id)` signature; re-implement job definitions in Dagster |
| **Data quality** | `DQGX` | `DQSodaPandera` | Same `validate(dataset)` contract; update rules engine internally |
| **Observability** | OTel→Prometheus/Grafana/Loki | OTel→SigNoz | Change collector endpoint; dashboards auto-adapt |
| **Secrets** | dotenv + GH secrets | Vault / cloud secrets | Change loader in ConfigService; keep `get_secret(key)` interface |

## 6. Implementation Timeline

### Phase 1: MVP Foundation (Weeks 1-4)
- [ ] Core adapter interfaces
- [ ] MVP implementations
- [ ] Basic infrastructure
- [ ] CI/CD pipeline

### Phase 2: MVP Features (Weeks 5-8)
- [ ] Trading strategies
- [ ] Backtesting engine
- [ ] LLM integration
- [ ] Basic frontend

### Phase 3: Scale Preparation (Weeks 9-12)
- [ ] Scale implementations
- [ ] Migration tools
- [ ] Performance testing
- [ ] Documentation

### Phase 4: Migration Tools (Weeks 13-16)
- [ ] Migration manager
- [ ] Rollback capabilities
- [ ] Validation tools
- [ ] Production deployment

## 7. Success Metrics

### MVP Success Criteria
- [ ] Deploy MVP in <8 weeks
- [ ] Support 100+ concurrent users
- [ ] Process 10K+ events/day
- [ ] 99.5% uptime

### Scale Migration Success Criteria
- [ ] Zero API changes during migration
- [ ] <5 minutes downtime per component
- [ ] 100% data integrity maintained
- [ ] Performance improvement >20%

## 8. Risk Mitigation

### Technical Risks
- **Adapter complexity**: Start with simple interfaces, evolve gradually
- **Performance overhead**: Benchmark adapters, optimize critical paths
- **Migration failures**: Comprehensive testing, automated rollback

### Business Risks
- **Vendor lock-in**: OSS-first approach, multiple provider support
- **Compliance gaps**: Built-in from day one, regular audits
- **Team expertise**: Training programs, gradual skill building

## 9. Future Considerations

### Beyond Scale Stack
- **Multi-cloud**: Kubernetes federation, cloud-agnostic storage
- **Edge computing**: Local model inference, reduced latency
- **AI/ML pipeline**: Automated feature engineering, model drift detection
- **Regulatory compliance**: GDPR, SOX, PCI-DSS support

---

*This PRD follows the [Cursor Agent Elite Protocol](https://github.com/cursor-ai/cursor-agent-elite-protocol) for professional-grade development standards.*
