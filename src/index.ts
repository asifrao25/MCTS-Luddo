/**
 * Luddo AI Engine - Express Server
 * MCTS-based AI for Luddo game
 *
 * Port: 3020
 */

import express from 'express';
import cors from 'cors';
import aiRoutes from './routes/ai.js';

const app = express();
const PORT = process.env.PORT || 3020;

// Middleware
app.use(cors({
  origin: [
    'http://localhost:3000',
    'http://localhost:5173',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5173',
    'https://luddo.asifrao.com',
  ],
  methods: ['GET', 'POST'],
  credentials: true,
}));

app.use(express.json({ limit: '1mb' }));

// Request logging
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);
  });
  next();
});

// Routes
app.use('/api/ai', aiRoutes);

// Root endpoint
app.get('/', (_req, res) => {
  res.json({
    service: 'Luddo AI Engine',
    version: '1.0.0',
    description: 'MCTS-based AI for intelligent move selection',
    endpoints: {
      'POST /api/ai/move': 'Get AI move decision',
      'GET /api/ai/health': 'Health check',
      'GET /api/ai/status': 'Service status',
    },
  });
});

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'healthy', uptime: process.uptime() });
});

// 404 handler
app.use((_req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Error handler
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('[Error]', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message,
  });
});

// Start server
app.listen(PORT, () => {
  console.log('');
  console.log('╔═══════════════════════════════════════════════╗');
  console.log('║       Luddo AI Engine - MCTS Service          ║');
  console.log('╠═══════════════════════════════════════════════╣');
  console.log(`║  Server running on port ${PORT}                 ║`);
  console.log('║  Model: MCTS + Heuristic Evaluation           ║');
  console.log('║  Time Budget: 5.5 seconds per move            ║');
  console.log('║  Iterations: Up to 50,000 per decision        ║');
  console.log('╚═══════════════════════════════════════════════╝');
  console.log('');
  console.log('Endpoints:');
  console.log('  POST /api/ai/move   - Get AI move decision');
  console.log('  GET  /api/ai/health - Health check');
  console.log('  GET  /api/ai/status - Service status');
  console.log('');
});

export default app;
