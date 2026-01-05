#!/usr/bin/env python
"""Run the Training Manager service."""

import uvicorn
from training_manager.config import PORT, HOST

if __name__ == "__main__":
    uvicorn.run(
        "training_manager.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )
