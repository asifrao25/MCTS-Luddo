module.exports = {
  apps: [
    {
      name: 'luddo-ai-engine',
      script: './dist/index.js',
      cwd: '/Users/m4-mac/mac-luddo',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'production',
        PORT: 3020
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: '/Users/m4-mac/mac-luddo/logs/error.log',
      out_file: '/Users/m4-mac/mac-luddo/logs/out.log',
      merge_logs: true
    },
    {
      name: 'luddo-neural-eval',
      script: 'neural_server.py',
      cwd: '/Users/m4-mac/mac-luddo/python',
      interpreter: '/Users/m4-mac/mac-luddo/python/venv/bin/python',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '300M',
      env: {
        PYTHONPATH: '/Users/m4-mac/mac-luddo/python'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: '/Users/m4-mac/mac-luddo/logs/neural-error.log',
      out_file: '/Users/m4-mac/mac-luddo/logs/neural-out.log',
      merge_logs: true
    },
    {
      name: 'luddo-training-manager',
      script: 'training_manager/run.py',
      cwd: '/Users/m4-mac/mac-luddo/python',
      interpreter: '/Users/m4-mac/mac-luddo/python/venv/bin/python',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        PYTHONPATH: '/Users/m4-mac/mac-luddo/python',
        PORT: 3022,
        DATABASE_PATH: '/Users/m4-mac/mac-luddo/python/training_manager/data/training_manager.db',
        DATA_DIR: '/Users/m4-mac/mac-luddo/data',
        MODELS_DIR: '/Users/m4-mac/mac-luddo/models'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: '/Users/m4-mac/mac-luddo/logs/training-manager-error.log',
      out_file: '/Users/m4-mac/mac-luddo/logs/training-manager-out.log',
      merge_logs: true
    }
  ]
};
