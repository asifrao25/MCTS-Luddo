module.exports = {
  apps: [
    {
      name: 'luddo-ai-engine',
      script: './dist/index.js',
      cwd: '/Users/m4-mac/luddo-ai-engine',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'production',
        PORT: 3020
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: '/Users/m4-mac/luddo-ai-engine/logs/error.log',
      out_file: '/Users/m4-mac/luddo-ai-engine/logs/out.log',
      merge_logs: true
    },
    {
      name: 'luddo-neural-eval',
      script: 'neural_server.py',
      cwd: '/Users/m4-mac/luddo-ai-engine/python',
      interpreter: '/Users/m4-mac/luddo-ai-engine/python/venv/bin/python',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '300M',
      env: {
        PYTHONPATH: '/Users/m4-mac/luddo-ai-engine/python'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: '/Users/m4-mac/luddo-ai-engine/logs/neural-error.log',
      out_file: '/Users/m4-mac/luddo-ai-engine/logs/neural-out.log',
      merge_logs: true
    }
  ]
};
