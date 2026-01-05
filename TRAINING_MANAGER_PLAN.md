# Luddo AI Training Management System - Implementation Plan

## Overview

New PM2 service (`luddo-training-manager` on port 3022) providing REST API + SSE for:
- **Simulation**: Generate training data with live progress
- **Training**: Manual MLX training with epoch-by-epoch progress
- **Benchmarking**: Model comparison with history
- **Dashboard**: Stats and active model info
- **System Metrics**: CPU, GPU, temperature monitoring

All data centralized on server; iOS/macOS apps are display clients.

---

## Architecture

```
Port 3020: luddo-ai-engine (Node.js)     [EXISTING]
Port 3021: luddo-neural-eval (Python)    [EXISTING]
Port 3022: luddo-training-manager (Python FastAPI) [NEW]
         └── SQLite DB for history
```

---

## API Endpoints

### Dashboard `/api/dashboard/*`
- GET `/stats` - Overview stats
- GET `/recent-activity` - Recent runs

### Simulation `/api/simulation/*`
- GET `/runs` - List runs
- GET `/runs/:id` - Run details
- POST `/start` - Start simulation
- POST `/stop/:id` - Stop simulation
- GET `/stream/:id` - SSE live progress

### Training `/api/training/*`
- GET `/runs` - List runs
- GET `/runs/:id` - Run with loss history
- POST `/start` - Start training
- POST `/stop/:id` - Stop training
- GET `/stream/:id` - SSE epoch progress
- GET `/data-sources` - Available datasets

### Benchmarks `/api/benchmark/*`
- GET `/runs` - List runs
- GET `/runs/:id` - Run details
- POST `/start` - Start benchmark
- POST `/stop/:id` - Stop benchmark
- DELETE `/runs/:id` - Delete result
- GET `/stream/:id` - SSE game progress

### Models `/api/models/*`
- GET `/` - List models
- GET `/:id` - Model details
- POST `/:id/activate` - Set active
- DELETE `/:id` - Delete model

### System `/api/system/*`
- GET `/health` - Health check
- POST `/reload-neural` - Reload neural server
- GET `/metrics` - System metrics snapshot
- GET `/metrics/stream` - SSE live metrics

---

## Implementation Phases

### Phase 1: Foundation
- [x] Create package structure
- [x] FastAPI app with CORS
- [x] SQLite database + migrations
- [x] Health endpoint
- [x] PM2 configuration

### Phase 2: Dashboard & Models
- [ ] Models registry CRUD
- [ ] Stats aggregation service
- [ ] Dashboard endpoints

### Phase 3: Simulation
- [ ] SSE manager
- [ ] Game engine (4-player)
- [ ] Feature extractor (all players)
- [ ] Simulation worker
- [ ] Simulation router

### Phase 4: Training
- [ ] Training worker with streaming
- [ ] Loss history tracking
- [ ] Training router

### Phase 5: Benchmarking
- [ ] Benchmark worker
- [ ] Model comparison logic
- [ ] Benchmark router

### Phase 6: System Metrics
- [ ] CPU, memory via psutil
- [ ] GPU/temp via subprocess
- [ ] Metrics streaming

### Phase 7: Polish
- [ ] Error handling
- [ ] Logging
- [ ] Testing
