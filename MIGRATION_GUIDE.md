# 🔄 Migration Guide: MVP → Scale

## Overview

This guide provides step-by-step instructions for migrating from the MVP architecture (Option A) to the Scale architecture (Option B) with **zero API changes** and minimal downtime.

## Migration Principles

1. **Zero API Changes**: All migrations happen behind service adapters
2. **Gradual Rollout**: Migrate component by component, not all at once
3. **Easy Rollback**: Built-in rollback capabilities for each component
4. **Validation**: Automated compatibility checks and performance validation
5. **Data Integrity**: 100% data preservation during migration

## Pre-Migration Checklist

### System Requirements
- [ ] Scale infrastructure is provisioned and tested
- [ ] All Scale adapters are implemented and tested
- [ ] Migration scripts are validated in staging
- [ ] Rollback procedures are tested
- [ ] Team is trained on Scale components
- [ ] Monitoring and alerting are configured

### Data Backup
- [ ] Full database backup completed
- [ ] Configuration files backed up
- [ ] User data exported
- [ ] API keys and secrets secured
- [ ] Audit logs preserved

### Communication
- [ ] Stakeholders notified of migration schedule
- [ ] Maintenance window communicated
- [ ] Support team briefed
- [ ] Rollback plan shared
- [ ] Success criteria defined

## Component Migration Map

### 1. Event Bus Migration (Redis → Kafka)

#### Pre-Migration
```bash
# Validate Kafka cluster
kafka-topics --bootstrap-server localhost:9092 --list

# Test Kafka connectivity
kafka-console-producer --bootstrap-server localhost:9092 --topic test
kafka-console-consumer --bootstrap-server localhost:9092 --topic test
```

#### Migration Steps
1. **Update Configuration**
   ```yaml
   # config/scale/event_bus.yaml
   event_bus:
     type: kafka
     bootstrap_servers: ["localhost:9092"]
     security_protocol: PLAINTEXT
     topics:
       trading_signals: trading_signals
       market_data: market_data
   ```

2. **Deploy Scale Adapter**
   ```bash
   # Deploy new Kafka-based event bus
   kubectl apply -f infrastructure/scale/kafka/
   
   # Verify deployment
   kubectl get pods -l app=event-bus-kafka
   ```

3. **Switch Traffic**
   ```bash
   # Update environment variable
   export EVENT_BUS_TYPE=kafka
   
   # Restart services
   kubectl rollout restart deployment/trading-engine
   ```

4. **Validation**
   ```bash
   # Check event processing
   kubectl logs -f deployment/event-bus-kafka
   
   # Verify no events lost
   python scripts/validate_event_migration.py
   ```

#### Rollback
```bash
# Revert to Redis
export EVENT_BUS_TYPE=redis
kubectl rollout restart deployment/trading-engine
```

### 2. LLM Service Migration (Groq/OpenAI → vLLM)

#### Pre-Migration
```bash
# Deploy vLLM service
kubectl apply -f infrastructure/scale/vllm/

# Test local model inference
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt", "max_tokens": 100}'
```

#### Migration Steps
1. **Update Configuration**
   ```yaml
   # config/scale/llm.yaml
   llm_service:
     type: vllm
     endpoint: http://vllm-service:8000
     models:
       default: mistral-7b-instruct
       trading: codellama-7b-instruct
   ```

2. **Deploy Scale Adapter**
   ```bash
   # Deploy vLLM adapter
   kubectl apply -f infrastructure/scale/llm/
   
   # Verify model loading
   kubectl logs -f deployment/llm-service-vllm
   ```

3. **Switch Traffic**
   ```bash
   # Update environment variable
   export LLM_SERVICE_TYPE=vllm
   
   # Restart services
   kubectl rollout restart deployment/llm-orchestrator
   ```

4. **Validation**
   ```bash
   # Test LLM responses
   python scripts/test_llm_migration.py
   
   # Verify response quality
   python scripts/validate_llm_quality.py
   ```

#### Rollback
```bash
# Revert to external LLM
export LLM_SERVICE_TYPE=groq
kubectl rollout restart deployment/llm-orchestrator
```

### 3. Database Migration (Supabase → Self-hosted Postgres)

#### Pre-Migration
```bash
# Provision Postgres cluster
kubectl apply -f infrastructure/scale/postgres/

# Test connectivity
kubectl exec -it postgres-0 -- psql -U postgres -d trading
```

#### Migration Steps
1. **Data Migration**
   ```bash
   # Export data from Supabase
   pg_dump $SUPABASE_URL > supabase_backup.sql
   
   # Import to new Postgres
   kubectl exec -i postgres-0 -- psql -U postgres -d trading < supabase_backup.sql
   ```

2. **Update Configuration**
   ```yaml
   # config/scale/database.yaml
   database:
     type: postgres
     host: postgres-service
     port: 5432
     database: trading
     username: postgres
     password: ${DB_PASSWORD}
   ```

3. **Switch Connection**
   ```bash
   # Update database URL
   export DATABASE_URL=postgresql://postgres:password@postgres-service:5432/trading
   
   # Restart services
   kubectl rollout restart deployment/trading-engine
   ```

4. **Validation**
   ```bash
   # Verify data integrity
   python scripts/validate_database_migration.py
   
   # Check performance
   python scripts/benchmark_database.py
   ```

#### Rollback
```bash
# Revert to Supabase
export DATABASE_URL=$SUPABASE_URL
kubectl rollout restart deployment/trading-engine
```

### 4. Backtesting Engine Migration (VectorBT → NautilusTrader)

#### Pre-Migration
```bash
# Deploy NautilusTrader service
kubectl apply -f infrastructure/scale/nautilus/

# Test basic functionality
python scripts/test_nautilus_basic.py
```

#### Migration Steps
1. **Update Configuration**
   ```yaml
   # config/scale/backtesting.yaml
   backtesting:
     engine: nautilus
     config:
       data_path: /data/market_data
       results_path: /data/backtest_results
       risk_limits:
         max_position_size: 100000
         max_drawdown: 0.1
   ```

2. **Deploy Scale Adapter**
   ```bash
   # Deploy NautilusTrader adapter
   kubectl apply -f infrastructure/scale/backtesting/
   
   # Verify deployment
   kubectl get pods -l app=backtesting-engine
   ```

3. **Switch Engine**
   ```bash
   # Update environment variable
   export BACKTESTING_ENGINE=nautilus
   
   # Restart services
   kubectl rollout restart deployment/backtesting-service
   ```

4. **Validation**
   ```bash
   # Run comparison backtests
   python scripts/compare_backtesting_engines.py
   
   # Verify result compatibility
   python scripts/validate_backtest_results.py
   ```

#### Rollback
```bash
# Revert to VectorBT
export BACKTESTING_ENGINE=vectorbt
kubectl rollout restart deployment/backtesting-service
```

## Migration Orchestration

### Automated Migration Script
```bash
#!/bin/bash
# scripts/migrate_to_scale.sh

set -e

echo "🚀 Starting MVP to Scale migration..."

# 1. Event Bus Migration
echo "📡 Migrating Event Bus..."
./scripts/migrate_event_bus.sh

# 2. LLM Service Migration
echo "🤖 Migrating LLM Service..."
./scripts/migrate_llm_service.sh

# 3. Database Migration
echo "🗄️ Migrating Database..."
./scripts/migrate_database.sh

# 4. Backtesting Engine Migration
echo "📊 Migrating Backtesting Engine..."
./scripts/migrate_backtesting.sh

echo "✅ Migration completed successfully!"
```

### Migration Validation
```bash
#!/bin/bash
# scripts/validate_migration.sh

echo "🔍 Validating migration..."

# Check all services
./scripts/check_service_health.sh

# Validate data integrity
./scripts/validate_data_integrity.sh

# Performance benchmarks
./scripts/run_performance_tests.sh

# API compatibility
./scripts/test_api_compatibility.sh

echo "✅ Validation completed!"
```

## Rollback Procedures

### Emergency Rollback
```bash
#!/bin/bash
# scripts/emergency_rollback.sh

echo "🚨 Emergency rollback initiated..."

# Rollback all components
./scripts/rollback_event_bus.sh
./scripts/rollback_llm_service.sh
./scripts/rollback_database.sh
./scripts/rollback_backtesting.sh

echo "🔄 Rollback completed!"
```

### Component-Specific Rollback
```bash
# Rollback specific component
./scripts/rollback_component.sh --component event_bus
./scripts/rollback_component.sh --component llm_service
./scripts/rollback_component.sh --component database
./scripts/rollback_component.sh --component backtesting
```

## Post-Migration Tasks

### 1. Performance Monitoring
```bash
# Monitor key metrics
kubectl top pods
kubectl logs -f deployment/monitoring

# Check application performance
python scripts/monitor_performance.py
```

### 2. Data Validation
```bash
# Verify data consistency
python scripts/validate_data_consistency.py

# Check for data loss
python scripts/check_data_loss.py
```

### 3. User Acceptance Testing
```bash
# Run user acceptance tests
pytest tests/integration/user_acceptance/

# Generate test report
python scripts/generate_uat_report.py
```

### 4. Documentation Update
```bash
# Update deployment docs
./scripts/update_deployment_docs.sh

# Update user guides
./scripts/update_user_guides.sh
```

## Troubleshooting

### Common Issues

#### Event Bus Issues
```bash
# Check Kafka connectivity
kafka-topics --bootstrap-server localhost:9092 --list

# Verify Redis fallback
redis-cli ping
```

#### LLM Service Issues
```bash
# Check vLLM service
kubectl logs -f deployment/llm-service-vllm

# Test model loading
curl -X GET http://localhost:8000/models
```

#### Database Issues
```bash
# Check Postgres connectivity
kubectl exec -it postgres-0 -- pg_isready

# Verify data integrity
python scripts/check_database_integrity.py
```

### Performance Issues
```bash
# Monitor resource usage
kubectl top nodes
kubectl top pods

# Check for bottlenecks
python scripts/analyze_performance.py
```

## Success Criteria

### Migration Success
- [ ] All components migrated successfully
- [ ] Zero API changes required
- [ ] Data integrity maintained
- [ ] Performance meets or exceeds MVP
- [ ] Zero downtime achieved

### Post-Migration Validation
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] User acceptance tests passed
- [ ] Documentation updated
- [ ] Team trained on new systems

## Support and Resources

### Documentation
- [Architecture Guide](docs/architecture/)
- [API Reference](docs/api/)
- [Deployment Guide](docs/deployment/)

### Tools and Scripts
- [Migration Scripts](migrations/scripts/)
- [Validation Tools](migrations/validation/)
- [Rollback Procedures](migrations/rollback/)

### Team Support
- **Technical Lead**: [@tech-lead](https://github.com/tech-lead)
- **DevOps Engineer**: [@devops-engineer](https://github.com/devops-engineer)
- **Data Engineer**: [@data-engineer](https://github.com/data-engineer)

---

*This migration guide follows the [Cursor Agent Elite Protocol](https://github.com/cursor-ai/cursor-agent-elite-protocol) for professional-grade development standards.*
